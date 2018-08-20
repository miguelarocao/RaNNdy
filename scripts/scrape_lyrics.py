"""
Scrapes lyrics from Genius for a given artist using Genius API.
Use https://docs.genius.com/ to create an api client.
App Name: RaNNdy
APP WEBSITE URL: https://github.com/miguelarocao/RaNNdy
Save client_id and access_token in a json file to use as credentials.
Run preprocess_sentences.py after scraping.
"""

from requests_oauthlib import OAuth2Session
from bs4 import BeautifulSoup
import requests
import argparse
import json

GENIUS_SEARCH_REQUEST = "https://api.genius.com/search?q={0}"
GENIUS_GET_SONG_REQUEST = "https://api.genius.com/artists/{0}/songs?per_page={1}&page={2}"


def search_artist(client, artist_query):
    """
    Searches for artists based on the query.
    :param client: [OAuth2Session] Client for API requests.
    :param artist_query: [String] Query.
    :return: [Dict] Artists matching the query and their IDs. {Artist Name: Artist ID}
    """
    api_response = client.get(GENIUS_SEARCH_REQUEST.format(artist_query)).json()
    search_hits = api_response['response']['hits']
    if not search_hits:
        print("No artist found!")
        return None
    return {hit['result']['primary_artist']['name']: hit['result']['primary_artist']['id'] for hit in search_hits}


def get_artist_songs(client, artist_id, songs_per_page=50):
    """
    Gets all the songs for a given artist.
    :param client: [OAuth2Session] Client for API requests.
    :param artist_id: [Int] Artist ID.
    :param songs_per_page: [Int] Songs per page. Maximum 50 (API limit).
    :return: [List[Dict]] All songs for the given artist. Keys per song: {title, id, url}
    """
    songs = []
    page = 1
    while (True):
        api_response = client.get(GENIUS_GET_SONG_REQUEST.format(artist_id, songs_per_page, page)).json()
        songs += [{key: song[key] for key in ['title', 'id', 'url']} for song in api_response['response']['songs']]
        page = api_response['response']['next_page']
        if page is None:
            break

    return songs


def get_song_lyrics(song_lyrics_url):
    """
    Gets the lyrics for the input song URL.
    :param song_lyrics_url: [String] URL for the desired scong to scrape.
    :return: [String] Song lyrics.
    """
    page = requests.get(song_lyrics_url)
    html = BeautifulSoup(page.text, "html.parser")
    return html.find("div", class_="lyrics").get_text()


def get_client(client_id, access_token, token_type='Bearer'):
    """
    Gets a an OAuth2 client.
    :param client_id: [String] Client ID.
    :param access_token: [String] Access token.
    :param token_type: [String] (Optional) Token type.
    :return: OAuth2Session
    """
    return OAuth2Session(client_id, token={'access_token': access_token, 'token_type': token_type})


def parse_lyrics(lyrics):
    """
    Parses lyrics. Removes empty lines, verse denotations, and splits lines.
    :param lyrics: [String] Lyrics.
    :return: [List[String]] Parsed lyrics.
    """
    output = []
    for line in lyrics.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line[0] == '[':
            continue
        output.append(line + '\n')

    return output


def main():
    # Define argument parsing
    parser = argparse.ArgumentParser(description='Scrapes web lyrics from www.genius.com')
    parser.add_argument('artist', help='What artist to gather lyrics for.', type=str)
    parser.add_argument('credentials', help='JSON containing Genius API credentials. (client_id, access_token)',
                        type=str)
    parser.add_argument('--output', help='Ouput file for song lyrics.', default='../data/lyrics.txt', type=str)

    args = parser.parse_args()

    # Load credentials
    with open(args.credentials, 'r') as f:
        credentials = json.load(f)

    client = get_client(credentials['client_id'], credentials['access_token'])
    artists = search_artist(client, args.artist)
    # print(artists)
    assert len(artists) == 1, f"There were multiple artists found! Please be more specific: {artists.keys()}"

    songs = get_artist_songs(client, artists[args.artist])

    print(f"{len(songs)} songs were found for {args.artist}.")

    with open(args.output, 'w') as f:
        for i, song in enumerate(songs):
            print(f"[{i}/{len(songs)}] Fetching song lyrics for '{song['title']}'...", end='')
            lyrics = get_song_lyrics(song['url'])
            print(f"Parsing...")
            f.writelines(parse_lyrics(lyrics))


if __name__ == '__main__':
    main()
