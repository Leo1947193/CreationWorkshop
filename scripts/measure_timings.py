import time
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

session_id = client.post('/api/v1/session/init').json()['session_id']
print('Session:', session_id)

start = time.time()
client.post('/api/v1/chat', json={'session_id': session_id, 'message': '海上城市由谁控制？'})
print('chat took', time.time()-start, 'seconds')

start = time.time()
client.post('/api/v1/analyze', json={'session_id': session_id})
print('analyze took', time.time()-start, 'seconds')
