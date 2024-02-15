from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        await asyncio.sleep(0.5)


# If your generator contains blocking operations such as time.sleep(),
# then define the generator function with normal `def`; or use `async def`,
# but run any blocking operations in external ThreadPool, etc. (see 2nd paragraph of this answer)
'''
import time

def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        time.sleep(0.5)
'''


@app.get('/')
async def main():
    return StreamingResponse(fake_data_streamer(), media_type='text/event-stream')
    # or, use:
    '''
    headers = {'X-Content-Type-Options': 'nosniff'}
    return StreamingResponse(fake_data_streamer(), headers=headers, media_type='text/plain')
    '''

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000);