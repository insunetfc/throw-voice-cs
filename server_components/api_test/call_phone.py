import requests

NIPA_BASE = "https://honest-trivially-buffalo.ngrok-free.app"

def make_call(phone):
    r = requests.post(f"{NIPA_BASE}/phone/call", data={"phone_number": phone})
    print(r.status_code, r.text)

def make_intro(name, phone):
    r = requests.post(
        f"{NIPA_BASE}/phone/generate-intro",
        data={"display_name": name, "phone_number": phone},
    )
    print(r.status_code, r.text)

if __name__ == "__main__":
    make_call("+821043876322")
    # make_intro("홍길동", "+821043876322")
