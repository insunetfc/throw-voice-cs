import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { UpdateCommand, DynamoDBDocumentClient } from "@aws-sdk/lib-dynamodb";

/**
 * Node.js 20 + AWS SDK v3 version of the original kvs_trigger.
 * - Uses async/await
 * - Replaces invokeAsync with InvokeCommand + InvocationType:"Event"
 * - Adds guards for missing Amazon Connect fields
 * - Adds try/catch logging around AWS calls
 */

// Reuse clients across invocations
const lambdaClient = new LambdaClient({});
const ddbClient = DynamoDBDocumentClient.from(new DynamoDBClient({}));

export const handler = async (event, context) => {
  console.log("Received event from Amazon Connect:", JSON.stringify(event));

  // Update DynamoDB (best effort; don't fail the whole handler if this errors)
  try {
    await updateDynamo(event);
  } catch (err) {
    console.error("updateDynamo failed (continuing):", err);
  }

  // Build payload safely with optional chaining + fallbacks
  const cd = event?.Details?.ContactData;
  const attrs = cd?.Attributes ?? {};
  const audio = cd?.MediaStreams?.Customer?.Audio ?? {};

  const payload = event?.eventType
    ? {
        inputFileName: "keepWarm.wav",
        connectContactId: "12b87d2b-keepWarm",
        transcriptionEnabled: "false",
      }
    : {
        streamARN: audio.StreamARN,
        startFragmentNum: audio.StartFragmentNumber,
        connectContactId: cd?.ContactId,
        transcriptionEnabled: attrs.transcribeCall === "true",
        saveCallRecording: attrs.saveCallRecording !== "false",
        languageCode: attrs.languageCode === "es-US" ? "es-US" : (attrs.languageCode || "en-US"),
        // default true for backwards compatibility
        streamAudioFromCustomer: attrs.streamAudioFromCustomer !== "false",
        streamAudioToCustomer: attrs.streamAudioToCustomer !== "false",
      };

  console.log("Trigger event passed to transcriberFunction:", JSON.stringify(payload));

  const params = {
    FunctionName: process.env.transcriptionFunction,
    InvocationType: "Event", // async, fire-and-forget (like invokeAsync)
    Payload: Buffer.from(JSON.stringify(payload)),
  };

  try {
    const data = await lambdaClient.send(new InvokeCommand(params));
    console.log("Invoked transcriptionFunction. StatusCode:", data?.StatusCode);
  } catch (err) {
    console.error("Error invoking transcriptionFunction:", err);
    // We still return Success to Connect to avoid breaking the flow.
  }

  return buildResponse();
};

function buildResponse() {
  return { lambdaResult: "Success" };
}

async function updateDynamo(event) {
  // You can override timezone via env var; default to original "America/New_York"
  process.env.TZ = process.env.TIMEZONE || "America/New_York";

  const cd = event?.Details?.ContactData;
  const customerPhoneNumber = cd?.CustomerEndpoint?.Address;
  const contactId = cd?.ContactId;
  const tableName = process.env.table_name;

  if (!tableName) {
    console.warn("updateDynamo skipped: process.env.table_name is not set");
    return;
  }
  if (!contactId) {
    console.warn("updateDynamo skipped: ContactId missing in event");
    return;
  }

  const currentTimeStamp = new Date().toString();
  const currentDate = new Date().toLocaleDateString();

  const paramsUpdate = {
    TableName: tableName,
    Key: { contactId },
    UpdateExpression: "SET customerPhoneNumber = :var1, callDate = :var2, callTimestamp = :var3",
    ExpressionAttributeValues: {
      ":var1": customerPhoneNumber ?? "unknown",
      ":var2": currentDate,
      ":var3": currentTimeStamp,
    },
  };

  try {
    const res = await ddbClient.send(new UpdateCommand(paramsUpdate));
    console.log("DynamoDB update succeeded:", JSON.stringify(res));
  } catch (err) {
    console.error("DynamoDB update failed:", err);
    throw err;
  }
}
