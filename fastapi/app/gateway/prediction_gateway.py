import json

from app.gateway.mysql_connection import MySQLConnection


def query_history(db: MySQLConnection):
    sql = '''\
        select
            i.id,
            m.tag,
            i.original_filename,
            i.resized_filename,
            p.result
        from images i
        left join predictions p on i.id = p.image_id
        left join models m on m.id = p.model_id
        order by i.id desc
        '''.strip()
    db.cur.execute(sql)
    rows = db.cur.fetchall()

    predictions = []
    for row in rows:
        prediction = {
            'imageId': row[0],
            'modelTag': row[1],
            'originalFilename': row[2],
            'resizedFilename': row[3],
            'result': row[4]
        }
        predictions.append(prediction)

    return predictions


def insert(db: MySQLConnection, model_id, image_id, result) -> None:
    sql = 'insert into predictions (model_id, image_id, result) values (%s, %s, %s)'
    db.cur.execute(sql, (model_id, image_id, json.dumps(result)))
    return db.cur.lastrowid
