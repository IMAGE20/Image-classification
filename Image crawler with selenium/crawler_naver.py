""" NAVER IMAGE CRAWLER WITH SELENIUM """

from selenium import webdriver
import os
import urllib.request

# 찾고자 하는 검색어를 url로 만들어 준다.
searchterm = '킬빌'
url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + searchterm

# chrome webdriver 사용하여 브라우저를 가져온다.
# 각자 경로에 따라 변경 필요 (chrome://version 에서 버전 확인 후 다운 받고, 파이썬 파일이 있는 폴더에 같이 두어야 함)
browser = webdriver.Chrome('C:/dev/final_project/python_code/crawler/chromedriver.exe')
browser.get(url)

# User-Agent를 통해 봇이 아닌 유저정보라는 것을 위해 사용
# Chrome 주소창에 chrome://version 접속 후 사용자 에이전트 값을 찾아서 변경
header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"}


# 이미지 카운트 (이미지 저장할 때 사용하기 위해서)
counter = 0
succounter = 0
print(os.path)

# 소스코드가 있는 경로에 '검색어' 폴더가 없으면 만들어준다.(이미지 저장 폴더를 위해서)
if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(1000):
    # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
    browser.execute_script("window.scrollBy(0,10000)")   # 화면상에서 0부터 10000까지 픽셀단위로 불러옴. 불러온 후 10001~ 이렇게 다음 불러옴.
    # JavaScript
    # scrollTo(x좌표, y좌표): 지정 위치에 스크롤
    # scrollBy(x좌표, y좌표): 상대 위치에 스크롤

# a 태그에서 class name이 entity인 것을 찾아온다
for x in browser.find_elements_by_tag_name('thumb _thumb'):
    # 구글 이미지 검색 결과와 달리 네이버의 경우 태그 이름이 명확하기 때문에
    # 코드를 간단히 하기 위해 xpath가 아닌 tag_name을 사용하는 걸로 바꿨습니다.
    # find_elements_by_tag_name vs. find_element_by_tag_name
    counter = counter + 1

    # 이미지 url
    imgurl = x.get_attribute('src')  # 사진 url 가져옴.
    # 이미지 확장자
    imgtype = 'jpg'
    # imgtype = x[x.rfind(".")+1:x.rfind("&")]   # rfind: 오른쪽부터 '.','&'을 찾아준다. 인덱스 슬라이싱

    print("Total Count:", counter)
    print("Succsessful Count:", succounter)
    print("URL:", imgurl)


    # naver 이미지를 읽고 저장한다.
    try:
        # urllib.request가 python3부터는 module이 되었기때문에, 이 기능을 수행하는 class Request를 호출합니다.
        req = urllib.request.Request(imgurl, headers=header)
        # urllib.request for opening and reading URLs
        # 그리고 headers parameter는 I AM NOT ROBOT임을 인증하는 파트로, 이를 dict type인 header 변수를 사용하도록 고쳤습니다.
        raw_img = urllib.request.urlopen(req).read()
        File = open(os.path.join(searchterm, searchterm + "_" + str(counter) + "." + imgtype), "wb")
        # wb: 쓰기모드 + 바이너리 모드, 바이트단위 데이터 기록에 사용
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
        print("can't get img")

print(succounter, "succesfully downloaded")
browser.close()



# 수정해야할 값들
#
# user_header
# : 현재 사용하고 있는 chrome의 버전에서 사용자 에이전트 정보를 입력합니다. chrome://version에서 확인합니다.
#
# set_title
# :폴더와 이미지 파일명을 결정합니다.
#
# driver
# : chrome://version에서 볼 수 있는 크롬의 버전과 동일한 chromedriver가 설치되어있어야하고, 그 프로그램의 경로를 지정합니다.
#
# number_of_page_down
# : 현재 화면을 page down 기능키로 몇 번 내릴 것인지 결정합니다(현재는 15이나, 50으로 늘리는 것을 추천)
#
# driver.find_element_by_xpath("""//*[@id="autopagerMore"]""").click()
# :Yahoo의 "더 보기" 버튼이므로, 문자열 //*[@id="autopagerMore"] 이 부분을 수정해야합니다.
# CTRL+SHIFT+C로 daum의 해당 버튼 위치를 찾아내고 우클릭->Copy->CopyXPath로 찾아보기