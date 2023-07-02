from Crypto import Random
from Crypto.PublicKey import RSA

# 偽亂數生成器
random_generator = Random.new().read

# 產生 1024 位元 RSA 金鑰
key = RSA.generate(1024, random_generator)

# RSA 私鑰
privateKey = key.export_key()
with open("./pytorch/keypairs/private.pem", "wb") as f:
    f.write(privateKey)

# RSA 公鑰
publicKey = key.publickey().export_key()
with open("./pytorch/keypairs/public.pem", "wb") as f:
    f.write(publicKey)

print('sk:',privateKey)

print('pk:',publicKey)