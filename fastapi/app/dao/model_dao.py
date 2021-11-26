from app.dao.dao_utils import MySQLConnection


class ModelDao:
    def find_by_tag(self, tag: str):
        with MySQLConnection() as db:
            sql = 'select id, tag, created_at from models where tag = %s'
            db.cur.execute(sql, (tag,))
            rows = db.cur.fetchall()

            if len(rows) == 0:
                return None

            row = rows[0]
            model = {
                'id': row[0],
                'tag': row[1],
                'created_at': row[2]
            }
            return model

    def insert(self, tag: str) -> None:
        with MySQLConnection() as db:
            sql = 'insert into models (tag) values (%s)'
            db.cur.execute(sql, (tag,))
            db.con.commit()
            return db.cur.lastrowid
