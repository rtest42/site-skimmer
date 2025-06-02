import puppeteer from 'puppeteer';

export async function ScrapeImage(startUrl, maxIterations, resWidth = 1920, resHeight = 1080){
    // Flags
    const resolution = {"width": resWidth, "height": resHeight};
    const heightFlag = resolution.height * 2;
    const timeoutFlag = 10000;
    const maxScrolls = 10;

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
            for (let numScrolls = 0; height < heightFlag && numScrolls < maxScrolls; ++numScrolls) {
                await page.evaluate(() => window.scrollBy(0, window.innerHeight));
                await new Promise(resolve => setTimeout(resolve, 1500));
                height = await page.evaluate(() => document.body.scrollHeight);
                // console.error(`height: ${height}`, `iterations: ${numScrolls}`);
            }

            // Collect image URLs
            const images = await page.evaluate(() =>
                Array.from(document.querySelectorAll('img')).map(img => img.src)
            );

            images.forEach(src => download_urls.add(src)); // Add image URLs to output
            
            // Collect more Pinterest links
            const links = await page.evaluate(() =>
                Array.from(document.querySelectorAll('a')).map(a => a.href)
            );
            const pinUrls = links.filter(link => link.includes("/pin"));
            to_visit.push(...pinUrls);

            visited_urls.add(url);
            iterations++;
        } catch (err) {
            console.error(`Failed on ${url}:`, err.message);
        }
    }

    console.log(JSON.stringify([...download_urls])); // Return value
    await browser.close();
}
