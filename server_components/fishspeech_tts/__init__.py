import gc
import queue
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest
import time


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
        load_embeddings: bool = True,
        embedding_path: str = "/home/work/VALL-E/audio_samples/fake_ref.pt",
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self.load_embeddings = load_embeddings
        self.embedding_path = embedding_path

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest):
        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
    
        req_ref_audio = getattr(req, "reference_audio", None)
        req_refs = getattr(req, "references", None)
    
        # Check if request actually brought a reference
        use_req_ref = (
            (req_ref_audio is not None)
            or (req_refs and len(req_refs) > 0)
            or (ref_id is not None)  # ← ref_id counts as bringing a ref!
        )
    
        # Try to load request-specific reference first
        if use_req_ref:
            if ref_id is not None:
                try:
                    prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)
                    logger.info(f"✅ Loaded reference by ID: {ref_id}")
                except Exception as e:
                    logger.warning(f"Failed to load ref_id {ref_id}: {e}")
            elif req_refs:
                prompt_tokens, prompt_texts = self.load_by_hash(req_refs, req.use_memory_cache)
                logger.info(f"✅ Loaded reference by hash")
            elif req_ref_audio is not None:
                prompt_tokens, prompt_texts = self.load_from_path(req_ref_audio, req.use_memory_cache)
                logger.info(f"✅ Loaded reference from path: {req_ref_audio}")
    
        # Fall back to default embedding only if no request reference was loaded
        if (not prompt_tokens or not prompt_texts) and self.load_embeddings:
            logger.info(f"📦 Loading default embedding from {self.embedding_path}")
            data = torch.load(self.embedding_path, map_location="cpu")
            prompt_tokens = data["prompt_tokens"]
            prompt_texts = data["prompt_texts"]
    
        # Final check
        if not prompt_tokens or not prompt_texts:
            logger.warning("⚠️ No prompt tokens/texts available, synthesis may fail or use model default")
    
        print(f'Done loading embeddings. Using {len(prompt_tokens)} prompt token(s).')
    
        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")
    
        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)
    
        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate
    
        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )
    
        segments = []
    
        while True:
            # Get the response from the LLAMA model
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break
    
            # Check the response type
            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )
    
            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                segment = self.get_audio_segment(result)
    
                if req.streaming:  # Used only by the API server
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )
                segments.append(segment)
            else:
                break
    
        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )
    
        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
