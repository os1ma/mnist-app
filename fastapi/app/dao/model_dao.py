from app.dao.dao_utils import MySQLConnection


def find_all(db: MySQLConnection):
    sql = 'select id, tag, created_at from models order by id'
    db.cur.execute(sql)
    rows = db.cur.fetchall()
    return [_row2dict(row) for row in rows]


def find_by_tag(db: MySQLConnection, tag: str):
    with MySQLConnection() as db:
        sql = 'select id, tag, created_at from models where tag = %s'
        db.cur.execute(sql, (tag,))
        rows = db.cur.fetchall()

        if len(rows) == 0:
            return None

        return _row2dict(rows[0])


def _row2dict(row):
    return {
        'id': row[0],
        'tag': row[1],
        'createdAt': row[2]
    }


def insert_if_not_exist(db: MySQLConnection, tag: str) -> None:
    sql = 'insert ignore into models (tag) values (%s)'
    db.cur.execute(sql, (tag,))
    return db.cur.lastrowid
