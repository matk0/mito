#!/usr/bin/env python3
"""
Advanced Paywall Detector
Sophisticated detection of paywall and premium content indicators.
"""

import re
from typing import Dict, List, Tuple, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class AdvancedPaywallDetector:
    """Advanced paywall and premium content detection"""
    
    def __init__(self):
        # Slovak paywall indicators with weights
        self.slovak_indicators = {
            # Strong indicators (high confidence)
            'prémium členov': 3.0,
            'premium členov': 3.0,
            'platených členov': 3.0,
            'iba pre členov': 3.0,
            'obsah pre predplatiteľov': 3.0,
            'obsah pre predplatitelov': 3.0,
            'members only': 3.0,
            'subscription required': 3.0,
            'premium content': 3.0,
            
            # Medium indicators
            'registrácia': 2.0,
            'registracia': 2.0,
            'prihlásiť sa': 2.0,
            'prihlasit sa': 2.0,
            'prihláste sa': 2.0,
            'prihlaste sa': 2.0,
            'predplatné': 2.0,
            'predplatne': 2.0,
            'webinár': 2.0,
            'webinar': 2.0,
            'pokračovanie článku': 2.0,
            'pokracovanie clanku': 2.0,
            'continue reading': 2.0,
            'read more': 2.0,
            'sign in': 2.0,
            'log in': 2.0,
            'register': 2.0,
            
            # Weak indicators
            'prihlásiť': 1.0,
            'prihlasit': 1.0,
            'registruj': 1.0,
            'vstúpiť': 1.0,
            'vstupte': 1.0,
            'join': 1.0,
            'become member': 1.0,
            'membership': 1.0,
        }
        
        # Truncation patterns
        self.truncation_patterns = [
            r'pokračovanie.*nájdete',
            r'pokracovanie.*najdete',
            r'zvyšok.*článku',
            r'zvysok.*clanku',
            r'čítať.*viac',
            r'citat.*viac',
            r'read.*more',
            r'continue.*reading',
            r'full.*article',
            r'kompletný.*článok',
            r'kompletny.*clanok',
        ]
        
        # Commercial indicators
        self.commercial_patterns = [
            r'kúpiť.*webinár',
            r'kupit.*webinar',
            r'zakúpiť.*kurz',
            r'zakupit.*kurz',
            r'buy.*course',
            r'purchase.*webinar',
            r'objednať.*školenie',
            r'objednat.*skolenie',
        ]
        
        # Content quality degradation indicators
        self.quality_degradation = [
            r'text.*pokračuje',
            r'text.*pokracuje',
            r'obsah.*pre.*členov',
            r'obsah.*pre.*clenov',
            r'content.*for.*members',
            r'exclusive.*content',
            r'exkluzívny.*obsah',
            r'exkluzivny.*obsah',
        ]
        
    def analyze_content(self, content: str, title: str = '', url: str = '') -> Dict:
        """Perform comprehensive paywall analysis"""
        result = {
            'paywall_detected': False,
            'confidence_score': 0.0,
            'indicators_found': [],
            'truncation_detected': False,
            'commercial_content': False,
            'content_quality_score': 1.0,
            'detailed_analysis': {},
            'recommendations': []
        }
        
        if not content:
            result['recommendations'].append("Empty content - cannot analyze")
            return result
            
        content_lower = content.lower()
        title_lower = title.lower() if title else ''
        
        # 1. Text-based indicator analysis
        text_analysis = self._analyze_text_indicators(content_lower, title_lower)
        result.update(text_analysis)
        
        # 2. Content structure analysis
        structure_analysis = self._analyze_content_structure(content)
        result['detailed_analysis']['structure'] = structure_analysis
        
        # 3. Truncation pattern detection
        truncation_analysis = self._detect_truncation_patterns(content_lower)
        result['truncation_detected'] = truncation_analysis['detected']
        result['detailed_analysis']['truncation'] = truncation_analysis
        
        # 4. Commercial content detection
        commercial_analysis = self._detect_commercial_content(content_lower)
        result['commercial_content'] = commercial_analysis['detected']
        result['detailed_analysis']['commercial'] = commercial_analysis
        
        # 5. Content quality assessment
        quality_analysis = self._assess_content_quality(content, content_lower)
        result['content_quality_score'] = quality_analysis['score']
        result['detailed_analysis']['quality'] = quality_analysis
        
        # 6. Final paywall determination
        final_analysis = self._determine_final_paywall_status(result)
        result.update(final_analysis)
        
        # 7. Generate specific recommendations
        result['recommendations'] = self._generate_detailed_recommendations(result)
        
        return result
        
    def _analyze_text_indicators(self, content_lower: str, title_lower: str) -> Dict:
        """Analyze text-based paywall indicators"""
        total_score = 0.0
        indicators_found = []
        
        # Check content
        for indicator, weight in self.slovak_indicators.items():
            count = content_lower.count(indicator)
            if count > 0:
                score_contribution = weight * count
                total_score += score_contribution
                indicators_found.append({
                    'text': indicator,
                    'weight': weight,
                    'count': count,
                    'location': 'content',
                    'score_contribution': score_contribution
                })
                
        # Check title
        for indicator, weight in self.slovak_indicators.items():
            if indicator in title_lower:
                score_contribution = weight * 0.5  # Lower weight for title
                total_score += score_contribution
                indicators_found.append({
                    'text': indicator,
                    'weight': weight,
                    'count': 1,
                    'location': 'title',
                    'score_contribution': score_contribution
                })
                
        # Normalize score (max score of 10 = 100% confidence)
        confidence = min(total_score / 10.0, 1.0)
        
        return {
            'confidence_score': confidence,
            'indicators_found': indicators_found,
            'paywall_detected': confidence > 0.3
        }
        
    def _analyze_content_structure(self, content: str) -> Dict:
        """Analyze content structure for paywall indicators"""
        analysis = {
            'total_length': len(content),
            'word_count': len(content.split()),
            'paragraph_count': len([p for p in content.split('\n') if p.strip()]),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'abrupt_ending': False,
            'suspicious_short_content': False
        }
        
        # Check for abrupt ending (common in paywall truncation)
        if content:
            last_sentences = content.split('.')[-3:]  # Last 3 sentences
            last_text = '.'.join(last_sentences).lower()
            
            # Look for truncation indicators in the end
            truncation_endings = [
                'pokračovanie', 'pokracovanie', 'continue', 'more',
                'register', 'registrácia', 'registracia'
            ]
            
            analysis['abrupt_ending'] = any(ending in last_text for ending in truncation_endings)
            
        # Check if content is suspiciously short
        if analysis['word_count'] < 100:
            analysis['suspicious_short_content'] = True
            
        return analysis
        
    def _detect_truncation_patterns(self, content_lower: str) -> Dict:
        """Detect content truncation patterns"""
        detected_patterns = []
        
        for pattern in self.truncation_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                detected_patterns.append({
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                })
                
        return {
            'detected': len(detected_patterns) > 0,
            'patterns_found': detected_patterns,
            'pattern_count': len(detected_patterns)
        }
        
    def _detect_commercial_content(self, content_lower: str) -> Dict:
        """Detect commercial/promotional content"""
        detected_patterns = []
        
        for pattern in self.commercial_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                detected_patterns.append({
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                })
                
        return {
            'detected': len(detected_patterns) > 0,
            'patterns_found': detected_patterns,
            'commercial_score': min(len(detected_patterns) / 3.0, 1.0)
        }
        
    def _assess_content_quality(self, content: str, content_lower: str) -> Dict:
        """Assess overall content quality"""
        quality_score = 1.0
        issues = []
        
        # Length-based quality assessment
        word_count = len(content.split())
        if word_count < 50:
            quality_score -= 0.5
            issues.append("Very short content")
        elif word_count < 200:
            quality_score -= 0.2
            issues.append("Short content")
            
        # Content coherence check
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(meaningful_sentences) < 3:
            quality_score -= 0.3
            issues.append("Few meaningful sentences")
            
        # Check for quality degradation indicators
        degradation_count = 0
        for pattern in self.quality_degradation:
            if re.search(pattern, content_lower, re.IGNORECASE):
                degradation_count += 1
                
        if degradation_count > 0:
            quality_score -= degradation_count * 0.2
            issues.append(f"Quality degradation indicators found: {degradation_count}")
            
        # Check content-to-fluff ratio
        content_words = len(content.split())
        
        # Count promotional/filler words
        filler_words = ['webinár', 'webinar', 'kurz', 'course', 'registrácia', 'registration']
        filler_count = sum(content_lower.count(word) for word in filler_words)
        
        if content_words > 0:
            filler_ratio = filler_count / content_words
            if filler_ratio > 0.1:  # More than 10% filler
                quality_score -= filler_ratio * 0.5
                issues.append(f"High promotional content ratio: {filler_ratio:.2f}")
                
        return {
            'score': max(0.0, quality_score),
            'issues': issues,
            'word_count': word_count,
            'meaningful_sentences': len(meaningful_sentences),
            'degradation_indicators': degradation_count
        }
        
    def _determine_final_paywall_status(self, analysis_result: Dict) -> Dict:
        """Determine final paywall status based on all analyses"""
        confidence = analysis_result.get('confidence_score', 0.0)
        truncation = analysis_result.get('truncation_detected', False)
        commercial = analysis_result.get('commercial_content', False)
        quality_score = analysis_result.get('content_quality_score', 1.0)
        
        # Weighted decision
        final_confidence = confidence
        
        # Boost confidence if truncation detected
        if truncation:
            final_confidence += 0.3
            
        # Boost confidence if commercial content detected
        if commercial:
            final_confidence += 0.2
            
        # Reduce confidence if quality is good
        if quality_score > 0.8:
            final_confidence *= 0.7
            
        # Cap at 1.0
        final_confidence = min(final_confidence, 1.0)
        
        # Final determination
        paywall_detected = final_confidence > 0.4
        
        # Confidence levels
        if final_confidence > 0.8:
            confidence_level = 'high'
        elif final_confidence > 0.5:
            confidence_level = 'medium'
        elif final_confidence > 0.2:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
            
        return {
            'paywall_detected': paywall_detected,
            'confidence_score': final_confidence,
            'confidence_level': confidence_level
        }
        
    def _generate_detailed_recommendations(self, analysis_result: Dict) -> List[str]:
        """Generate detailed recommendations based on analysis"""
        recommendations = []
        
        confidence = analysis_result.get('confidence_score', 0.0)
        paywall_detected = analysis_result.get('paywall_detected', False)
        truncation = analysis_result.get('truncation_detected', False)
        commercial = analysis_result.get('commercial_content', False)
        quality_score = analysis_result.get('content_quality_score', 1.0)
        
        if paywall_detected:
            if confidence > 0.8:
                recommendations.append("CRITICAL: Strong paywall detected - exclude from GraphRAG")
            elif confidence > 0.5:
                recommendations.append("MAJOR: Likely paywall detected - manual review recommended")
            else:
                recommendations.append("MINOR: Possible paywall - investigate further")
                
        if truncation:
            recommendations.append("Content appears truncated - attempt re-scraping with authentication")
            
        if commercial:
            recommendations.append("High commercial content - consider filtering promotional text")
            
        if quality_score < 0.5:
            recommendations.append("Poor content quality - exclude from knowledge base")
        elif quality_score < 0.7:
            recommendations.append("Medium content quality - review for GraphRAG suitability")
            
        if not paywall_detected and quality_score > 0.7:
            recommendations.append("Content appears suitable for GraphRAG")
            
        if not recommendations:
            recommendations.append("No significant issues detected")
            
        return recommendations
        
    def batch_analyze(self, articles: List[Dict]) -> List[Dict]:
        """Analyze multiple articles for paywall content"""
        results = []
        
        for i, article in enumerate(articles):
            logger.debug(f"Analyzing article {i+1}/{len(articles)} for paywall content")
            
            content = article.get('content', '')
            title = article.get('title', '')
            url = article.get('url', '')
            
            analysis = self.analyze_content(content, title, url)
            analysis['article_index'] = i
            analysis['source_article'] = {
                'url': url,
                'title': title,
                'original_word_count': article.get('word_count', 0)
            }
            
            results.append(analysis)
            
        return results
        
    def get_paywall_summary(self, analysis_results: List[Dict]) -> Dict:
        """Generate summary statistics for paywall analysis"""
        total_articles = len(analysis_results)
        paywall_detected = sum(1 for r in analysis_results if r.get('paywall_detected', False))
        
        confidence_distribution = {
            'high': sum(1 for r in analysis_results if r.get('confidence_level') == 'high'),
            'medium': sum(1 for r in analysis_results if r.get('confidence_level') == 'medium'),
            'low': sum(1 for r in analysis_results if r.get('confidence_level') == 'low'),
            'very_low': sum(1 for r in analysis_results if r.get('confidence_level') == 'very_low')
        }
        
        truncation_detected = sum(1 for r in analysis_results if r.get('truncation_detected', False))
        commercial_content = sum(1 for r in analysis_results if r.get('commercial_content', False))
        
        avg_quality_score = sum(r.get('content_quality_score', 0) for r in analysis_results) / total_articles if total_articles > 0 else 0
        
        return {
            'total_articles': total_articles,
            'paywall_detected_count': paywall_detected,
            'paywall_percentage': (paywall_detected / total_articles * 100) if total_articles > 0 else 0,
            'confidence_distribution': confidence_distribution,
            'truncation_detected_count': truncation_detected,
            'commercial_content_count': commercial_content,
            'average_quality_score': avg_quality_score,
            'graphrag_suitable_count': sum(1 for r in analysis_results 
                                         if not r.get('paywall_detected', False) and 
                                            r.get('content_quality_score', 0) > 0.7)
        }