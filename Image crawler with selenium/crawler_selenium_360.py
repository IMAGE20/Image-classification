""" 360 IMAGE CRAWLER WITH SELENIUM """

from selenium import webdriver
import os
import urllib.request

# 찾고자 하는 검색어를 url로 만들어 준다.
searchterm = '恙虫病焦痂'
url = "https://image.so.com/i?q=" + searchterm + "&src=tab_www"

# chrome webdriver 사용하여 브라우저를 가져온다.
browser = webdriver.Chrome('C:/dev/final_project/python_code/crawler/chromedriver.exe')  # 각자 경로에 따라 변경
browser.get(url)

# User-Agent를 통해 봇이 아닌 유저정보라는 것을 위해 사용
# Chrome 주소창에 chrome://version 접속 후 사용자 에이전트 값을 찾아서 변경
header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"}

# 이미지 카운트 (이미지 저장할 때 사용하기 위해서)
counter = 0
succounter = 0

# 소스코드가 있는 경로에 '검색어' 폴더가 없으면 만들어준다.(이미지 저장 폴더를 위해서)
if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(1000):
    # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
    browser.execute_script("window.scrollBy(0,10000)")
    # JavaScript
    # scrollTo(x좌표, y좌표): 지정 위치에 스크롤
    # scrollBy(x좌표, y좌표): 상대 위치에 스크롤

for img in browser.find_elements_by_tag_name('img'):
    # 구글 이미지 검색 결과와 달리 360의 경우 태그 이름이 명확하기 때문에 코드를 간단히 하기 위해 xpath가 아닌 tag_name을 사용하는 걸로 바꿨습니다.
    # find_elements_by_tag_name vs. find_element_by_tag_name

    counter = counter + 1
    # 이미지 url
    imgurl = img.get_attribute('src')
    # 이미지 확장자
    imgtype = imgurl[imgurl.rfind(".") + 1:]

    print("Total Count:", counter)
    print("Succsessful Count:", succounter)
    print("URL:", imgurl)

    # 360 이미지를 읽고 저장한다.
    try:
        req = urllib.request.Request(imgurl, headers=header)
        raw_img = urllib.request.urlopen(req).read()
        File = open(os.path.join(searchterm, searchterm + "_" + str(counter) + "." + imgtype), "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
        print("can't get img")

print(succounter, "succesfully downloaded")
browser.close()