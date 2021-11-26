from app.dao.dao_utils import MySQLConnection


class ImageDao:
    def find_all(self, tag: str):
        with MySQLConnection() as db:
            sql = 'select id, original_image_path, preprocessed_image_path, created_at from models order by id'
            db.cur.execute(sql)
            rows = db.cur.fetchall()

            images = []
            for row in rows:
                image = {
                    'id': row[0],
                    'original_image_path': row[1],
                    'preprocessed_image_path': row[2],
                    'created_at': row[3]
                }
                images.append(image)

            return images

    def insert(self, original_image_path: str, preprocessed_image_path: str) -> int:
        with MySQLConnection() as db:
            sql = 'insert into images (original_image_path, preprocessed_image_path) values (%s, %s)'
            db.cur.execute(sql, (original_image_path, preprocessed_image_path))
            db.con.commit()
            return db.cur.lastrowid
