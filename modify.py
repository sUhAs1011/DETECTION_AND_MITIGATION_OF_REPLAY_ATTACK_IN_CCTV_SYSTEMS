import sqlite3
conn = sqlite3.connect("hashing_module/db/hash_store.db")
cursor = conn.cursor()
cursor.execute("SELECT phash FROM frame_hashes LIMIT 5;")
print(cursor.fetchall())
conn.close()
