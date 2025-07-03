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
  LOGIN_URL = "#{BASE_URL}/sign-in/"
  OUTPUT_DIR = './scraped_data'
  
  def initialize(username = nil, password = nil)
    @scraped_articles = []
    @failed_urls = []
    @username = username
    @password = password
    @cookies = {}
    @authenticated = false
    setup_output_directory
  end
  
  def scrape_all
    puts "Starting to scrape #{BLOG_URL}"
    
    # Login if credentials provided
    if @username && @password
      puts "Attempting to login..."
      if login
        puts "✓ Successfully logged in"
        @authenticated = true
      else
        puts "✗ Login failed, continuing without authentication"
      end
    else
      puts "No credentials provided, scraping public content only"
    end
    
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
  
  def login
    begin
      # First, get the login page to extract form details
      login_page_response = fetch_url_raw(LOGIN_URL)
      return false unless login_page_response
      
      doc = Nokogiri::HTML(login_page_response.body)
      
      # Look for WordPress login form
      form = doc.css('form#loginform, form[action*="wp-login"], form[action*="sign-in"]').first
      if form
        action_url = form['action']
        action_url = "#{BASE_URL}#{action_url}" if action_url&.start_with?('/')
        action_url ||= "#{BASE_URL}/wp-login.php"
      else
        # Fallback to standard WordPress login
        action_url = "#{BASE_URL}/wp-login.php"
      end
      
      # Prepare login data
      login_data = {
        'log' => @username,
        'pwd' => @password,
        'wp-submit' => 'Log In',
        'redirect_to' => BLOG_URL,
        'testcookie' => '1'
      }
      
      # Submit login form
      uri = URI(action_url)
      response = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
        request = Net::HTTP::Post.new(uri)
        request['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        request['Content-Type'] = 'application/x-www-form-urlencoded'
        request['Referer'] = LOGIN_URL
        
        # Add existing cookies
        request['Cookie'] = cookies_to_string if @cookies.any?
        
        request.body = URI.encode_www_form(login_data)
        
        http.request(request)
      end
      
      # Extract cookies from response
      extract_cookies_from_response(response)
      
      # Check if login was successful
      if response.code == '302' || response.code == '200'
        # Verify authentication by checking a protected page
        test_response = fetch_url_raw(BLOG_URL)
        return test_response && !test_response.body.include?('PLATENÝCH ČLENOV')
      end
      
      false
    rescue => e
      puts "Login error: #{e.message}"
      false
    end
  end
  
  def extract_cookies_from_response(response)
    response.get_fields('Set-Cookie')&.each do |cookie|
      cookie_parts = cookie.split(';').first.split('=', 2)
      if cookie_parts.length == 2
        @cookies[cookie_parts[0]] = cookie_parts[1]
      end
    end
  end
  
  def cookies_to_string
    @cookies.map { |k, v| "#{k}=#{v}" }.join('; ')
  end
  
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
    response = fetch_url_raw(url)
    return nil unless response
    
    if response.code == '200'
      response.body
    else
      puts "HTTP #{response.code} for #{url}"
      nil
    end
  end
  
  def fetch_url_raw(url)
    uri = URI(url)
    
    Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
      request = Net::HTTP::Get.new(uri)
      request['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
      
      # Add cookies if we have them
      if @cookies.any?
        request['Cookie'] = cookies_to_string
      end
      
      response = http.request(request)
      
      # Extract cookies from response
      extract_cookies_from_response(response)
      
      response
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
  
  # Check for credentials as command line arguments
  username = ARGV[0]
  password = ARGV[1]
  
  if username && password
    puts "Using provided credentials for authentication"
    scraper = BlogScraper.new(username, password)
  else
    puts "No credentials provided. Set them as arguments: ruby blog_scraper.rb username password"
    puts "Continuing without authentication (public content only)"
    scraper = BlogScraper.new
  end
  
  scraper.scrape_all
end
