import os 
import json
import re
# from utils import get_urlhash
import syslog
from bs4 import BeautifulSoup


class PageQualFeature:
    def __init__(self):
        self.pageout = {} 
        self.pagein = {}
        
    def build(self,root_path):        
        print("[START] BUILD")
        for p in os.listdir(root_path):
            if p[0] !='.':
                for f in os.listdir(root_path+p):
                    with open(root_path+'/'+str(p)+'/'+str(f)) as page_file:
                        print(f'[START] READING {f}')
                        page = json.load(page_file)
                        pages = BeautifulSoup(page['content'])
                        syslog.syslog(f'[END] READING {f}')
                        syslog.syslog(f'[START] Extract HyperLinks')
                        hls = self._extract_hyperlinks(pages)
                        syslog.syslog(f'[END] Extract HyperLinks')
                        syslog.syslog(f'[START] Build pageRankDB')
                        self._build_pagerankdb(page['url'],hls)
                        syslog.syslog(f'[END] Extract pageRankDB')
        print("[END] BUILD")
        print("[START] Cal pagerank " )
        self.cal_pagerank()
        print("[END] Cal pagerank " )
        #self.save_pagefeat()
        #print("[END] Save all features" )
    
    
    def is_valid(self, url):
        from urllib.parse import urlparse
        # Decide whether to crawl this url or not. 
        # If you decide to crawl it, return True; otherwise return False.
        # There are already some conditions that return False.
        try:
            parsed = urlparse(url)
            if parsed.scheme not in set(["http", "https"]):
                return False

            if re.match(
                r".*\.(css|js|bmp|gif|jpe?g|ico"
                + r"|png|tiff?|mid|mp2|mp3|mp4"
                + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
                + r"|ps|eps|tex|ppt|pptx|ppsx|doc|docx|xls|xlsx|names"
                + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
                + r"|epub|dll|cnf|tgz|sha1"
                + r"|thmx|mso|arff|rtf|jar|csv"
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower()):
                return False
            # print("Matching {} and the regex matching is {}".
            #     format(parsed.netloc.lower(), re.match(r".*\.(stat|ics|cs)\.uci\.edu", parsed.netloc.lower())))
            # return re.match(r".*\.(stat|ics|cs)\.uci\.edu", parsed.netloc.lower())
            return self.filter_to_allow_subdomains(parsed)
        except TypeError:
            print ("TypeError for ", parsed)
            raise
            
    def filter_to_allow_subdomains(self, parsed):
        return  re.match(r".*\.(stat|ics|cs)\.uci\.edu", parsed.netloc.lower()) or (
                    re.match(r"today\.uci\.edu", parsed.netloc.lower()) and
                    re.match(r"department/information_computer_sciences/.*", parsed.path.lower())
                )

    def _extract_hyperlinks(self, soup):
        return list(set([l['href'] for l in soup.find_all('a', href=True) if len(l['href']) > 3 ]))
    
    def _build_pagerankdb(self,url,hyperlinks):
        #--[TODO] change it to urlhash later
        #urlhash = get_urlhash(url)
        urlhash = url
        hyperhash = hyperlinks
        #hyperhash = [get_urlhash(h) for h in hyperlinks]
        self.pageout[urlhash] = hyperhash
        for h in hyperhash:
            if h not in self.pagein:
                self.pagein[h] = [urlhash]
            else:
                self.pagein[h].append(urlhash)
                
    def cal_pagerank(self,tol = 1.0e-7):
        PR = {p:1/len(self.pageout) for p in self.pageout}
        i = 0 
        while True:
            i +=1 
            new_PR= {}
            for p in PR:
                newrank = PR[p]
                if p in self.pagein:
                    newrank = 1/len(self.pageout) + 0.9*sum([PR[ip]/len(self.pageout[ip]) for ip in self.pagein[p] ])
                new_PR[p] = newrank
            err = sum([abs(new_PR[n] - PR[n]) for n in PR])
            if err < len(PR)*tol:
                print(f"Converged : iter{i}")
                self.PR= PR
                return 
            PR = new_PR 
            
    def save_pagefeat(self,URL_to_docID_map, path='./page_quality_features.json'):
        page_quality_features = {}
        for p in self.pageout:
            pid = URL_to_docID_map[p]
            page_quality_features[pid] = {}
            page_quality_features[pid]["pageout"] = len(self.pageout[p])
        for p in self.pageout:
            pid = URL_to_docID_map[p]
            if p in self.pagein:
                page_quality_features[pid]["pagein"] = len(self.pagein[p])
            else:
                page_quality_features[pid]["pagein"] = 0
        for p in self.PR:
            pid = URL_to_docID_map[p]
            page_quality_features[pid]["pagerank"] = self.PR[p]
            
        with open(path, "w") as outfile:
            json.dump(page_quality_features, outfile)
        