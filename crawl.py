import requests
import os

class crawler():
    def __init__(self):
        self.url = 'https://api-staging.jfbapp.cn/quiz/'
        self.path = './hupu'
        self.image_path = './hupu/images'
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)'}

    def go(self, begin=5, end=122):
        score = {}
        for i in range(begin, end):
            try:
                res = requests.get(self.url + str(i), headers=self.headers).json()
                print("get {}".format(res['quiz']['title']))
                questions = res['questions']
                for question in questions:
                    image_id = question['id']
                    image_url = question['image']
                    avg_score = question['avgScore']
                    score[image_id] = avg_score
                    for _ in range(2):
                        try:
                            image_name = '{}/images/{}.jpg'.format(self.path, image_id)
                            image_res = requests.get(image_url)
                            with open(image_name, "wb") as f:
                                f.write(image_res.content)
                            break
                        except Exception as e:
                            print(e)
            except Exception as e:
                print(e)
                continue
        with open('{}/{}'.format(self.path, 'score.json'), 'w') as f:
            f.write(str(score).replace("'", '"'))

if __name__ == "__main__":
    obj = crawler()
    obj.go()
