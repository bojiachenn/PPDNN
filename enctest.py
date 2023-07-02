from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
 
# 加密
message = "Hello,This is RSA加密"
pk = RSA.importKey(open("./pytorch/keypairs/public.pem").read())
print(pk)
cipher = Cipher_pkcs1_v1_5.new(pk)     #建立用於執行pkcs1_v1_5加密或解密的密碼
cipher_text = base64.b64encode(cipher.encrypt(message.encode('utf-8')))
print(cipher_text.decode('utf-8'))