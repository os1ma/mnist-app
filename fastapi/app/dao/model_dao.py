from app.dao.dao_utils import MySQLConnection


class ModelDao:
    def find_all(self):
        with MySQLConnection() as db:
            sql = 'select id, tag, created_at from models order by id'
            db.cur.execute(sql)
            rows = db.cur.fetchall()
            return [self._row2dict(row) for row in rows]

    def find_by_tag(self, tag: str):
        with MySQLConnection() as db:
            sql = 'select id, tag, created_at from models where tag = %s'
            db.cur.execute(sql, (tag,))
            rows = db.cur.fetchall()

            if len(rows) == 0:
                return None

            return self._row2dict(rows[0])

    def _row2dict(self, row):
        return {
            'id': row[0],
            'tag': row[1],
            'createdAt': row[2]
        }

    def insert_if_not_exist(self, tag: str) -> None:
        with MySQLConnection() as db:
            sql = 'insert ignore into models (tag) values (%s)'
            db.cur.execute(sql, (tag,))
            db.con.commit()
            return db.cur.lastrowid
