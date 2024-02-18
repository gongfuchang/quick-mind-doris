from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import time


app = FastAPI()


def data_streamer():
    yield """                
{"query": "What are the other two generations of leptons?", "rid": "Bilg5uVPyqgzX6OPPmfDT", "contexts": [{"id": "https://api.bing.microsoft.com/api/v7/#WebPages.0", "contractualRules": [{"_type": "ContractualRules/LicenseAttribution", "targetPropertyName": "snippet", "targetPropertyIndex": 0, "mustBeCloseToContent": true, "license": {"name": "CC-BY-SA", "url": "http://creativecommons.org/licenses/by-sa/3.0/"}, "licenseNotice": "Text under CC-BY-SA license"}], "name": "Generation (particle physics) - Wikipedia", "url": "https://en.wikipedia.org/wiki/Generation_(particle_physics)", "isFamilyFriendly": true, "displayUrl": "https://en.wikipedia.org/wiki/Generation_(particle_physics)", "snippet": "In particle physics, a generation or family is a division of the elementary particles.Between generations, particles differ by their flavour quantum number and mass, but their electric and strong interactions are identical.. There are three generations according to the Standard Model of particle physics. Each generation contains two types of leptons and two types of quarks.", "dateLastCrawled": "2024-02-10T22:01:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4808602796163142&mkt=en-US&setlang=en-US&w=uJ5nmx-N9TGKswNAE4FKuuavdPUjpGD4", "language": "en", "isNavigational": false}]}
    """
    time.sleep(2)
    yield """                
__RELATED_QUESTIONS__
[{"question": "What are the leptons in the second generation?"}, {"question": "What are the leptons in the third generation?"}]    

    """
    time.sleep(1)
    yield """                
__LLM_RESPONSE__

        """
    for i in range(10):
        yield f"_{i}_".encode("utf-8")
        time.sleep(0.2)


@app.get('/query')
async def main():
    return EventSourceResponse(data_streamer(), media_type='text/event-stream')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000);