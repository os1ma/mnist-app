from app.dao.dao_utils import MySQLConnection


class ImageDao:
    def insert(self, original_image_path: str, preprocessed_image_path: str) -> int:
        with MySQLConnection() as db:
            sql = 'insert into images (original_image_path, preprocessed_image_path) values (%s, %s)'
            db.cur.execute(sql, (original_image_path, preprocessed_image_path))
            db.con.commit()
            return db.cur.lastrowid
