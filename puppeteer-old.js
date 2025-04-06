import puppeteer from 'puppeteer-core';
import fs from 'fs';
import { execSync } from 'child_process';

// Function to find Edge
const findEdge = () => {
  const paths = [
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe'
  ];
  return paths.find(path => fs.existsSync(path)) || null;
};

// Function to find Chrome
const findChrome = () => {
  const paths = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
  ];
  return paths.find(path => fs.existsSync(path)) || null;
};

// Function to find Firefox
const findFirefox = () => {
  const paths = [
    'C:/Program Files/Mozilla Firefox/firefox.exe',
    'C:/Program Files (x86)/Mozilla Firefox/firefox.exe'
  ];
  return paths.find(path => fs.existsSync(path)) || null;
};

// Function to find Safari (MacOS only)
const findSafari = () => {
  try {
    const path = execSync('defaults read /Applications/Safari.app/Contents/Info CFBundleExecutable', { encoding: 'utf8' }).trim();
    return path ? '/Applications/Safari.app/Contents/MacOS/Safari' : null;
  } catch {
    return null;
  }
};

const getBrowserPath = () => {
  return findEdge() || findChrome() || findFirefox() || findSafari() || (() => {
    throw new Error("No supported browser found");
  })();
};

const main = async (websites) => {
  const browser = await puppeteer.launch({ executablePath: getBrowserPath(), headless: false});

  const page = await browser.newPage();


  // Go to the Pinterest search page
  await page.goto('https://www.pinterest.com/search/pins/?q=pants&rs=typed', { waitUntil: 'domcontentloaded',});

  // Wait for the page to load (adjust timeout as needed)
  await page.waitForSelector('a');

  // Extract links using page.evaluate() and DOM querying
  const pinterestLinks = await page.evaluate(() => {
    const links = document.querySelectorAll('a');
    return Array.from(links)
      .map(link => link.href)
      .filter(href => href.includes('https://www.pinterest.com/pin/'));
  });

  console.log(pinterestLinks);  // Outputs all the links

  await browser.close();

  /*
  // Array of promises for each website
  const promises = websites.map(async (website) => {
    const page = await browser.newPage();
    await page.goto(website);
    const allImages = await page.evaluate(() => {
      return Array.from(document.images, img => img.src);
    });
    // Get all links
    const allLinks = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('a'), a => a.href);
    });

    console.log(allImages);  // Will be returned as subprocess takes stdin
    console.log("LINKS");
    console.log(allLinks);   // Logs the links
    await page.close();
  });

  // Wait for all promises to resolve
  await Promise.all(promises);
  await browser.close();
  */
};

// main(fs.readFileSync(process.argv[2], 'utf8').trim().split('\n'));
main(fs.readFileSync('test.txt', 'utf8').trim().split('\n'));