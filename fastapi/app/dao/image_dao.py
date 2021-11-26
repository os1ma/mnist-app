from app.dao.dao_utils import MySQLConnection


class ImageDao:
    def find_all(self):
        with MySQLConnection() as db:
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

    def insert(self, original_filename: str, resized_filename: str) -> int:
        with MySQLConnection() as db:
            sql = 'insert into images (original_filename, resized_filename) values (%s, %s)'
            db.cur.execute(sql, (original_filename, resized_filename))
            db.con.commit()
            return db.cur.lastrowid
