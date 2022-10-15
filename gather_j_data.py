
import requests, pandas, csv
import numpy as np
from bs4 import BeautifulSoup

NUM_GAMES = 7000

def html_table_to_list(html):
  html = html.find_all(class_="score_positive")
  ints = []
  for elem in html:
    ints.append(int(elem.text.replace('$', '').replace(',', '')))
  return ints

data = []
for game_id in range(NUM_GAMES):
  if game_id % 100 == 0 and game_id > 0:
    print(f"\r{game_id} games searched")
  url = "https://www.j-archive.com/showgame.php?game_id="
  page = requests.get(url+str(game_id))
  soup = BeautifulSoup(page.content, 'html.parser')
  try:
    head = soup.find('h3', text='Scores at the end of the Double Jeopardy! Round:')
    scores_before_html = head.next_sibling.next_sibling
    scores_before = html_table_to_list(scores_before_html)

    head = soup.find('h3', text='Final scores:')
    scores_after_html = head.next_sibling.next_sibling
    scores_after = html_table_to_list(scores_after_html)

    if len(scores_before) == 2:
      # if there are only two players in Final Jeopardy!, assume the third player is 0
      scores_before.append(0)
      scores_after.append(0)
    else:
      # if there is only one player (apparently this happens?), discard this datum
      continue
      
    data.append([scores_before, scores_after])
    
  except AttributeError:
    continue

data = np.array(data)
NUM_GAMES = len(data)
df = pandas.DataFrame(data.reshape((NUM_GAMES, -1)))
df.to_csv('/content/drive/MyDrive/Final_J_data.csv', header=False, index=False)