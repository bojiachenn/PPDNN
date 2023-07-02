from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
 
# 解密
cipher_text = input("Ciphertext:")
encrypt_text = cipher_text.encode('utf-8')
rsakey = RSA.importKey(open("./pytorch/keypairs/private.pem").read())
cipher = Cipher_pkcs1_v1_5.new(rsakey)      #建立用於執行pkcs1_v1_5加密或解密的密碼
text = cipher.decrypt(base64.b64decode(encrypt_text), "解密失敗")
print(text.decode('utf-8'))