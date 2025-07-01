#!/usr/bin/env ruby

require 'nokogiri'
require 'net/http'
require 'uri'
require 'json'
require 'fileutils'
require 'time'

class BlogScraper
  BASE_URL = 'https://jaroslavlachky.sk'
  BLOG_URL = "#{BASE_URL}/blog/"
  OUTPUT_DIR = './scraped_data'
  
  def initialize
    @scraped_articles = []
    @failed_urls = []
    setup_output_directory
  end
  
  def scrape_all
    puts "Starting to scrape #{BLOG_URL}"
    
    page = 1
    total_scraped = 0
    
    loop do
      puts "\n--- Scraping page #{page} ---"
      article_urls = scrape_article_urls_from_page(page)
      
      if article_urls.empty?
        puts "No more articles found. Stopping."
        break
      end
      
      puts "Found #{article_urls.length} articles on page #{page}"
      
      article_urls.each do |url|
        begin
          article_data = scrape_article(url)
          if article_data
            @scraped_articles << article_data
            total_scraped += 1
            puts "✓ Scraped: #{article_data[:title][0..50]}..."
            
            # Save individual article
            save_article_to_file(article_data)
            
            # Small delay to be respectful
            sleep(0.5)
          else
            puts "✗ Failed to scrape: #{url}"
            @failed_urls << url
          end
        rescue => e
          puts "✗ Error scraping #{url}: #{e.message}"
          @failed_urls << url
        end
      end
      
      page += 1
      
      # Safety break - adjust if needed
      break if page > 50
    end
    
    puts "\n=== Scraping Complete ==="
    puts "Total articles scraped: #{total_scraped}"
    puts "Failed URLs: #{@failed_urls.length}"
    
    save_summary
    save_failed_urls if @failed_urls.any?
  end
  
  private
  
  def setup_output_directory
    FileUtils.mkdir_p(OUTPUT_DIR)
    FileUtils.mkdir_p("#{OUTPUT_DIR}/articles")
  end
  
  def scrape_article_urls_from_page(page_num)
    url = page_num == 1 ? BLOG_URL : "#{BLOG_URL}page/#{page_num}/"
    
    begin
      response = fetch_url(url)
      return [] unless response
      
      doc = Nokogiri::HTML(response)
      
      # Try multiple selectors to find article links
      article_links = []
      
      # Common WordPress selectors
      selectors = [
        'article h2 a',
        '.entry-title a',
        '.post-title a',
        'h2.entry-title a',
        '.article h2 a'
      ]
      
      selectors.each do |selector|
        links = doc.css(selector)
        if links.any?
          article_links = links.map { |link| normalize_url(link['href']) }
          break
        end
      end
      
      # Fallback: look for any links that seem like blog posts
      if article_links.empty?
        all_links = doc.css('a[href]')
        article_links = all_links.map { |link| link['href'] }
                                .select { |href| href && href.include?(BASE_URL) && !href.include?('/blog/') && href.count('/') >= 4 }
                                .map { |href| normalize_url(href) }
                                .uniq
      end
      
      article_links.uniq
    rescue => e
      puts "Error fetching page #{page_num}: #{e.message}"
      []
    end
  end
  
  def scrape_article(url)
    response = fetch_url(url)
    return nil unless response
    
    doc = Nokogiri::HTML(response)
    
    # Extract title
    title = extract_title(doc)
    return nil unless title
    
    # Extract content
    content = extract_content(doc)
    return nil if content.empty?
    
    # Extract metadata
    date = extract_date(doc)
    excerpt = extract_excerpt(doc)
    
    {
      url: url,
      title: title,
      content: content,
      date: date,
      excerpt: excerpt,
      scraped_at: Time.now.iso8601,
      word_count: content.split.length
    }
  end
  
  def extract_title(doc)
    selectors = [
      'h1.entry-title',
      '.entry-title',
      'h1',
      'title'
    ]
    
    selectors.each do |selector|
      element = doc.css(selector).first
      if element
        title = element.text.strip
        return title unless title.empty?
      end
    end
    
    nil
  end
  
  def extract_content(doc)
    selectors = [
      '.pf-content .entry_content',
      '.blog_entry_content',
      '.entry_content',
      '.post-content',
      '.article-content',
      '.content',
      'article .entry-content',
      'main article'
    ]
    
    selectors.each do |selector|
      element = doc.css(selector).first
      if element
        # Remove unwanted elements - including specific Slovak blog elements
        element.css('script, style, nav, header, footer, aside, .comments, .social-share, .in_share_element, .fb-like, .twitter-like, .printfriendly, .ve_form_element, form, .mw_social_icons_container, .related_posts').remove
        
        content = element.text.strip
        return content if content.length > 100  # Minimum content length
      end
    end
    
    # Fallback: try to get main content area
    main_content = doc.css('main, .main, #main, #content, .content-area, .blog-content').first
    if main_content
      main_content.css('script, style, nav, header, footer, aside, .comments, .social-share, .in_share_element, .fb-like, .twitter-like, .printfriendly, form').remove
      content = main_content.text.strip
      return content if content.length > 100
    end
    
    ''
  end
  
  def extract_date(doc)
    selectors = [
      'time[datetime]',
      '.published',
      '.entry-date',
      '.post-date',
      '[class*="date"]'
    ]
    
    selectors.each do |selector|
      element = doc.css(selector).first
      if element
        date_str = element['datetime'] || element.text.strip
        begin
          return Time.parse(date_str).iso8601
        rescue
          # Continue to next selector
        end
      end
    end
    
    nil
  end
  
  def extract_excerpt(doc)
    selectors = [
      '.excerpt',
      '.entry-summary',
      '.post-excerpt'
    ]
    
    selectors.each do |selector|
      element = doc.css(selector).first
      return element.text.strip if element
    end
    
    # Fallback: first paragraph of content
    first_p = doc.css('.entry-content p, .post-content p, .content p').first
    first_p ? first_p.text.strip[0..200] : ''
  end
  
  def fetch_url(url)
    uri = URI(url)
    
    Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
      request = Net::HTTP::Get.new(uri)
      request['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      
      response = http.request(request)
      
      if response.code == '200'
        response.body
      else
        puts "HTTP #{response.code} for #{url}"
        nil
      end
    end
  rescue => e
    puts "Network error for #{url}: #{e.message}"
    nil
  end
  
  def normalize_url(url)
    return nil unless url
    
    if url.start_with?('/')
      "#{BASE_URL}#{url}"
    elsif url.start_with?('http')
      url
    else
      "#{BASE_URL}/#{url}"
    end
  end
  
  def save_article_to_file(article)
    filename = article[:title].gsub(/[^\w\s-]/, '').gsub(/\s+/, '_')[0..50]
    filepath = "#{OUTPUT_DIR}/articles/#{filename}.json"
    
    File.write(filepath, JSON.pretty_generate(article))
  end
  
  def save_summary
    summary = {
      total_articles: @scraped_articles.length,
      scraping_completed_at: Time.now.iso8601,
      total_words: @scraped_articles.sum { |a| a[:word_count] },
      articles_by_year: @scraped_articles.group_by { |a| a[:date] ? Date.parse(a[:date]).year : 'unknown' }.transform_values(&:count)
    }
    
    File.write("#{OUTPUT_DIR}/scraping_summary.json", JSON.pretty_generate(summary))
    puts "\nSummary saved to #{OUTPUT_DIR}/scraping_summary.json"
  end
  
  def save_failed_urls
    File.write("#{OUTPUT_DIR}/failed_urls.txt", @failed_urls.join("\n"))
    puts "Failed URLs saved to #{OUTPUT_DIR}/failed_urls.txt"
  end
end

# CLI interface
if __FILE__ == $0
  puts "Slovak Blog Scraper"
  puts "==================="
  
  scraper = BlogScraper.new
  scraper.scrape_all
end
