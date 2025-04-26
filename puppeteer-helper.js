import puppeteer from 'puppeteer';

async function ScrapeImage(startUrl, maxIterations){
    // Flags
    let resolution = {"width": 2560, "height": 1440};
    let heightFlag = 3000;
    let timeoutFlag = 10000;

    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({ width: resolution.width, height: resolution.height });
        
    let iterations = 0;
    let download_urls = new Set();
    let visited_urls = new Set();
    let to_visit = [startUrl];

    while (to_visit.length > 0 && iterations <= maxIterations) {
        const url = to_visit.pop();
        if (visited_urls.has(url)) continue; // Skip URL that has already been seen

        try {
            await page.goto(url, { waitUntil: 'domcontentloaded' });
            await page.waitForSelector('img', { visible: true, timeout: timeoutFlag });
            
            // Scroll to load images
            let height = 0;
            while (height < heightFlag) {
                previousHeight = await page.evaluate(() => document.body.scrollHeight);
                await page.evaluate(() => window.scrollBy(0, window.innerHeight));
                await new Promise(resolve => setTimeout(resolve, 1500));
                height = await page.evaluate(() => document.body.scrollHeight);
            }

            // Collect image URLs
            const images = await page.evaluate(() =>
                Array.from(document.querySelectorAll('img')).map(img => img.src)
            );

            images.forEach(src => download_urls.add(src)); // Add image URLs to output

            const links = await page.evaluate(() =>
                Array.from(document.querySelectorAll('a')).map(a => a.href)
            );
            const pinUrls = links.filter(link => link.includes("/pin"));
            to_visit.push(...pinUrls);

            visited_urls.add(url);
            iterations++;
        } catch (err) {
            console.warn(`Failed on ${url}: ${err.message}`);
        }

        console.log(JSON.stringify([...download_urls])); // Return value
        await browser.close();
    }
}

export { ScrapeImage };
