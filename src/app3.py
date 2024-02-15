from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import time


app = FastAPI()


def data_streamer():
    for i in range(10):
        yield f"_{i}_".encode("utf-8")
        time.sleep(1)


@app.get('/')
async def main():
    return EventSourceResponse(data_streamer(), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000);