from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


async def fake_data_streamer():
    yield """                
{"query": "What are the other two generations of leptons?", "rid": "Bilg5uVPyqgzX6OPPmfDT", "contexts": [{"id": "https://api.bing.microsoft.com/api/v7/#WebPages.0", "contractualRules": [{"_type": "ContractualRules/LicenseAttribution", "targetPropertyName": "snippet", "targetPropertyIndex": 0, "mustBeCloseToContent": true, "license": {"name": "CC-BY-SA", "url": "http://creativecommons.org/licenses/by-sa/3.0/"}, "licenseNotice": "Text under CC-BY-SA license"}], "name": "Generation (particle physics) - Wikipedia", "url": "https://en.wikipedia.org/wiki/Generation_(particle_physics)", "isFamilyFriendly": true, "displayUrl": "https://en.wikipedia.org/wiki/Generation_(particle_physics)", "snippet": "In particle physics, a generation or family is a division of the elementary particles.Between generations, particles differ by their flavour quantum number and mass, but their electric and strong interactions are identical.. There are three generations according to the Standard Model of particle physics. Each generation contains two types of leptons and two types of quarks.", "dateLastCrawled": "2024-02-10T22:01:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4808602796163142&mkt=en-US&setlang=en-US&w=uJ5nmx-N9TGKswNAE4FKuuavdPUjpGD4", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.1", "contractualRules": [{"_type": "ContractualRules/LicenseAttribution", "targetPropertyName": "snippet", "targetPropertyIndex": 1, "mustBeCloseToContent": true, "license": {"name": "CC-BY-SA", "url": "http://creativecommons.org/licenses/by-sa/3.0/"}, "licenseNotice": "Text under CC-BY-SA license"}], "name": "Lepton - Wikipedia", "url": "https://en.wikipedia.org/wiki/Lepton", "isFamilyFriendly": true, "displayUrl": "https://en.wikipedia.org/wiki/Lepton", "snippet": "In particle physics, a lepton is an elementary particle of half-integer spin (spin 1 \u2044 2) that does not undergo strong interactions. Two main classes of leptons exist: charged leptons (also known as the electron-like leptons or muons), and neutral leptons (better known as neutrinos).Charged leptons can combine with other particles to form various composite particles such as atoms and ...", "dateLastCrawled": "2024-02-15T01:32:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4543573252713745&mkt=en-US&setlang=en-US&w=h1IkBxBEpyQSLZCH9YUgxPTeKl_C44un", "language": "en", "isNavigational": false, "richFacts": [{"label": {"text": "Generation"}, "items": [{"text": "1st, 2nd, 3rd"}], "hint": {"text": "BASE:GENERICFACT"}}, {"label": {"text": "Composition"}, "items": [{"_type": "Properties/Link", "text": "Elementary particle", "url": "https://www.bing.com/search?q=Elementary+particle+wikipedia"}], "hint": {"text": "BASE:GENERICFACTWITHLINK"}}, {"label": {"text": "Statistics"}, "items": [{"_type": "Properties/Link", "text": "Fermionic", "url": "https://www.bing.com/search?q=Fermionic+wikipedia"}], "hint": {"text": "BASE:GENERICFACTWITHLINK"}}]}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.2", "name": "The Standard Model | CERN", "url": "https://home.cern/science/physics/standard-model", "isFamilyFriendly": true, "displayUrl": "https://home.cern/science/physics/standard-model", "snippet": "These particles occur in two basic types called quarks and leptons. Each group consists of six particles, which are related in pairs, or \u201cgenerations\u201d. The lightest and most stable particles make up the first generation, whereas the heavier and less-stable particles belong to the second and third generations.", "dateLastCrawled": "2024-02-16T13:19:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=5007614402179185&mkt=en-US&setlang=en-US&w=LDGaqge1f1rlM1qffb1WEOw-UwLw97LD", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.3", "name": "Leptons - HyperPhysics", "url": "http://hyperphysics.phy-astr.gsu.edu/hbase/Particles/lepton.html", "isFamilyFriendly": true, "displayUrl": "hyperphysics.phy-astr.gsu.edu/hbase/Particles/lepton.html", "snippet": "Leptons. Leptons and quarks are the basic building blocks of matter, i.e., they are seen as the \"elementary particles\". There are six leptons in the present structure, the electron, muon, and tau particles and their associated neutrinos.The different varieties of the elementary particles are commonly called \"flavors\", and the neutrinos here are considered to have distinctly different flavor.", "dateLastCrawled": "2024-02-16T15:03:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4890220062983615&mkt=en-US&setlang=en-US&w=Rm4jGgguHVuv0JeNpudiHsi0Ze21MzQB", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.4", "name": "Origin of lepton/quark generations? - Physics Stack Exchange", "url": "https://physics.stackexchange.com/questions/45849/origin-of-lepton-quark-generations", "about": [{"_type": "CreativeWork", "aggregateRating": {"ratingValue": 0, "reviewCount": 7}}], "isFamilyFriendly": true, "displayUrl": "https://physics.stackexchange.com/questions/45849", "snippet": "This condition is satisfied by the Standard Model particles: 3 (2/3 - 1/3) + (0 - 1) = 0. Which tells us that the quark and lepton doublets in a generation really are paired in a non-trivial way. If they weren't paired up, the theory would most likely be inconsistent. Share. Cite.", "dateLastCrawled": "2024-02-11T18:54:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4756221380608114&mkt=en-US&setlang=en-US&w=2f6-weZuWGBy4Ura1mlXejAn1z2Ec8ND", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.5", "name": "The mystery of particle generations | symmetry magazine", "url": "https://www.symmetrymagazine.org/article/august-2015/the-mystery-of-particle-generations?language_content_entity=und", "isFamilyFriendly": true, "displayUrl": "https://www.symmetrymagazine.org/article/august-2015/the-mystery-of-particle-generations", "snippet": "The next generations. The Standard Model is a menu listing all of the known fundamental particles: particles that cannot be broken down into constituent parts. It distinguishes between the fermions, which are particles of matter, and the bosons, which carry forces. The matter particles include six quarks and six leptons.", "dateLastCrawled": "2024-02-13T10:02:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4532492238848163&mkt=en-US&setlang=en-US&w=qRhN7JxNNyENI4bXF8YbVvIMYDUFvvqF", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.6", "name": "Lepton | Encyclopedia.com", "url": "https://www.encyclopedia.com/science-and-technology/physics/physics/lepton", "datePublished": "2018-05-09T00:00:00.0000000", "datePublishedDisplayText": "May 9, 2018", "isFamilyFriendly": true, "displayUrl": "https://www.encyclopedia.com/science-and-technology/physics/physics/lepton", "snippet": "The set of leptons can be arranged into three generations, as shown in Table 1. There is an electron, muon, and tau lepton family. Each generation has two particles and two antiparticles, where the antiparticles have the same mass as the particle but opposite quantum numbers. Each force has an associated charge.", "dateLastCrawled": "2024-02-08T18:03:00.0000000Z", "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4577056814413056&mkt=en-US&setlang=en-US&w=Oc3hLnxcgeghZ51PYi6Tbt5_z0Dxh2MZ", "language": "en", "isNavigational": false}, {"id": "https://api.bing.microsoft.com/api/v7/#WebPages.7", "name": "Leptons: The elementary particles explained | Space", "url": "https://www.space.com/leptons-facts-explained", "thumbnailUrl": "https://www.bing.com/th?id=OIP.fa4QRIEZSrpi-POB-XKc-AAAAA&w=80&h=80&c=1&pid=5.1", "datePublished": "2023-01-20T00:00:00.0000000", "datePublishedDisplayText": "Jan 20, 2023", "isFamilyFriendly": true, "displayUrl": "https://www.space.com/leptons-facts-explained", "snippet": "Leptons are elementary particles, which means that they are not made from any smaller particles. There are six known types of lepton (12 if you count their anti-particles). Three of these are ...", "dateLastCrawled": "2024-02-16T02:04:00.0000000Z", "primaryImageOfPage": {"thumbnailUrl": "https://www.bing.com/th?id=OIP.fa4QRIEZSrpi-POB-XKc-AAAAA&w=80&h=80&c=1&pid=5.1", "width": 80, "height": 80, "imageId": "OIP.fa4QRIEZSrpi-POB-XKc-AAAAA"}, "cachedPageUrl": "http://cc.bingj.com/cache.aspx?q=What+are+the+other+two+generations+of+leptons%3f&d=4846389924228269&mkt=en-US&setlang=en-US&w=sQ8uVLSBX5Sb4nHYXBfLa2xe9xZjp_PO", "language": "en", "isNavigational": false}]}

        """
    await asyncio.sleep(5)

    yield """                
__RELATED_QUESTIONS__
[{"question": "What are the leptons in the second generation?"}, {"question": "What are the leptons in the third generation?"}]    

        """
    await asyncio.sleep(10)
    yield """                
__LLM_RESPONSE__

            """
    for i in range(100):
        yield f"_{i}_".encode("utf-8")
        await asyncio.sleep(0.2)


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


@app.get('/query')
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