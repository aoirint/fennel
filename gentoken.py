import os
from uuid import uuid4
import jwt

from dotenv import load_dotenv
load_dotenv()

JWT_SECRET = os.environ['JWT_SECRET']

payload = {
  'sub': str(uuid4())
}
print(payload)

token_string = jwt.encode(payload=payload, key=JWT_SECRET, algorithm='HS256')

print(token_string)
