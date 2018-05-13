import scrapy
from scrapy.crawler import CrawlerProcess


class HitsSpider(scrapy.Spider):
    name = "hits"

    start_urls = [
        'https://www.billboard.com/archive/charts'
    ]

    def parse(self, response):

        answer = response.xpath('.//li[@class="year-list__decade__dropdown__item"]')
        for selector in answer:
            year = selector.xpath('.//a/text()').extract_first()
            link = selector.xpath('.//a/@href').extract_first()
            self.log(year)

            request_details = scrapy.Request('https://www.billboard.com'+link+'/hot-100', self.parse_table)
            request_details.meta['data'] = {
                'year': year,
                'link_year': link
            }
            yield request_details

    def parse_table(self, response):
        data = response.meta['data']
        answer = response.xpath('//table[contains(@class,"archive-table")]/tbody/tr')

        for selector in answer:
            data['title'] = selector.xpath('.//td[2]/text()').extract_first()
            data['artist'] = selector.xpath('.//td[3]/text()').extract_first()

            yield data


process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_FORMAT': 'json',
    'FEED_URI': './data/billboard_extract.json',
    'FEED_EXPORT_ENCODING': 'utf-8',
    'COOKIE_ENABLED': False,
    'DOWNLOAD_DELAY': 6
})

process.crawl(HitsSpider)
process.start()  # the script will block here until the crawling is finished