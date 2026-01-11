# Placeholder classifier Lambda.
# Replace with a real SageMaker endpoint call or HF pipeline.
import json, os, random, datetime as dt

LABELS = ['neutral','hate','misinformation','emergency']

def handler(event, context):
    # event is expected to be one record (dict) or list of records
    def score_one(rec):
        random.seed(rec.get('id','seed'))
        label = random.choice(LABELS)
        confidence = round(0.5 + random.random()/2, 2)
        rec.update({
            'label': label,
            'confidence': confidence,
            'classified_at': dt.datetime.utcnow().isoformat() + 'Z',
            'model': 'distilbert-demo-stub'
        })
        return rec

    if isinstance(event, list):
        out = [score_one(r) for r in event]
    else:
        out = score_one(event)
    return { 'status': 'ok', 'data': out }
