from app.gateway.mysql_connection import MySQLConnection


def find_all(db: MySQLConnection):
    sql = 'select id, original_filename, resized_filename, created_at from images order by id'
    db.cur.execute(sql)
    rows = db.cur.fetchall()

    images = []
    for row in rows:
        image = {
            'id': row[0],
            'originalFilename': row[1],
            'resizedFilename': row[2],
            'createdAt': row[3]
        }
        images.append(image)

    return images


def insert(db: MySQLConnection, original_filename: str, resized_filename: str) -> int:
    sql = 'insert into images (original_filename, resized_filename) values (%s, %s)'
    db.cur.execute(sql, (original_filename, resized_filename))
    return db.cur.lastrowid
