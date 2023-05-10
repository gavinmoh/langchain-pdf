import hashlib

# hashing a file and use the hash as key for caching the embeddings
def sha256sum(file):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    while n := file.readinto(mv):
        h.update(mv[:n])
    return h.hexdigest()