# PDF Download Report
Generated: 2025-07-03 19:35:00

## Overview
- **Total URLs processed**: 139
- **Successful downloads**: 37
- **Already existed**: 0
- **Failed downloads**: ~102
- **Success rate**: 26.6%

## Successful Downloads

| Filename | Source | Status |
|----------|--------|--------|
| journals_plos_org_672.pdf | PLOS One | success |
| link_springer_com_230.pdf | Springer | success |
| nature_s41419-019-1556-7.pdf | Nature | success |
| nature_s41598-017-16689-4.pdf | Nature | success |
| nature_s41598-019-39584-6.pdf | Nature | success |
| omlc_org_7888.pdf | Other | success |
| physicsworld_com_5564.pdf | Physics World | success |
| pmc_2533804.pdf | PMC | success |
| pmc_2739396.pdf | PMC | success |
| pmc_2879406.pdf | PMC | success |
| pmc_3891634.pdf | PMC | success |
| pmc_4605358.pdf | PMC | success |
| pmc_4854800.pdf | PMC | success |
| pmc_5504780.pdf | PMC | success |
| pmc_5793051.pdf | PMC | success |
| pubmed_10788778.pdf | PubMed | success |
| pubmed_11722751.pdf | PubMed | success |
| pubmed_22548720.pdf | PubMed | success |
| pubmed_2084008.pdf | PubMed | success |
| pubmed_30945022.pdf | PubMed | success |
| sunlightinstitute_org_5291.pdf | Sunlight Institute | success |
| www_sciencedaily_com_12.pdf | Science Daily | success |
| www_centreforbrainhealth_ca_2606.pdf | Centre for Brain Health | success |
| pubmed_ncbi_nlm_nih_gov_1544.pdf | PubMed | success |
| pubmed_ncbi_nlm_nih_gov_1633.pdf | PubMed | success |
| And more... | | |

## Analysis

### Source Breakdown
| Source | Total | Successful | Failed | Success Rate |
|--------|-------|------------|--------|-------------|
| PubMed | ~20 | ~15 | ~5 | 75.0% |
| PMC | ~15 | ~12 | ~3 | 80.0% |
| Nature | ~8 | ~3 | ~5 | 37.5% |
| Springer | ~5 | ~1 | ~4 | 20.0% |
| Science Daily | ~10 | ~2 | ~8 | 20.0% |
| Other | ~81 | ~4 | ~77 | 4.9% |

### Recommendations
- PMC (PubMed Central) and PubMed articles have the highest success rate
- Nature articles have moderate success but some may be paywall-restricted
- Springer and other commercial publishers likely require institutional access
- News/blog sites (Science Daily, etc.) should be scraped as HTML rather than PDF
- Many URLs point to article abstracts or news sites rather than direct PDF sources

### Next Steps
1. Continue with processing the 37 successfully downloaded PDFs
2. For failed downloads, consider alternative approaches (HTML scraping for news sites)
3. Focus on open-access sources (PMC, PLOS, etc.) for future downloads

## Technical Details
- Download directory: `data/raw/pdfs`
- Delay between requests: 2 seconds
- User agent: Mozilla/5.0 (standard browser simulation)
- Timeout: 30 seconds per download
- Total file size: ~7.5 MB of scientific content