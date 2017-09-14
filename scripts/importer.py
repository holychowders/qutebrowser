#!/usr/bin/env python3
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:

# Copyright 2014-2017 Claude (longneck) <longneck@scratchbook.ch>
# Copyright 2014-2017 Florian Bruhin (The Compiler) <mail@qutebrowser.org>

# This file is part of qutebrowser.
#
# qutebrowser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qutebrowser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qutebrowser.  If not, see <http://www.gnu.org/licenses/>.


"""Tool to import data from other browsers.

Currently only importing bookmarks from Netscape Bookmark files is supported.
"""


import argparse


def main():
    args = get_args()
    bookmark_types = []
    output_format = ''
    if args.search_query or args.search_output:
        bookmark_types = ['search']
        if args.newconfig:
            output_format = 'ncsearch'
        else:
            output_format = 'search'
    else:
        if args.bookmark_output:
            output_format = 'bookmark'
        elif args.quickmark_output:
            output_format = 'quickmark'
        if args.bookmark_query:
            bookmark_types.append('bookmark')
        if args.keyword_query:
            bookmark_types.append('keyword')
    if not bookmark_types:
        bookmark_types = ['bookmark','keyword']
    if not output_format:
        output_format = 'quickmark'

    if args.browser in ['chromium', 'firefox', 'ie']:
        import_netscape_bookmarks(args.bookmarks,bookmark_types,output_format)


def get_args():
    """Get the argparse parser."""
    parser = argparse.ArgumentParser(
        epilog="To import bookmarks from Chromium, Firefox or IE, "
               "export them to HTML in your browsers bookmark manager. "
               "By default, this script will output in a quickmarks format.")
    parser.add_argument('browser', help="Which browser? (chromium, firefox)",
                        choices=['chromium', 'firefox', 'ie'],
                        metavar='browser')
    parser.add_argument('-b', help="Output in bookmark format.",
                        dest='bookmark_output', action='store_true',
                        default=False, required=False)
    parser.add_argument('-q', help="Output in quickmark format (default).",
                        dest='quickmark_output', action='store_true',
                        default=False,required=False)
    parser.add_argument('-s', help="Output search engine format",
                        dest='search_output', action='store_true',
                        default=False,required=False)
    parser.add_argument('--newconfig', help="Output search engine format for new config.py format",
                        default=False,action='store_true',required=False)
    parser.add_argument('-S', help="Import search engines",
                        dest='search_query', action='store_true',
                        default=False,required=False)
    parser.add_argument('-B', help="Import plain bookmarks (no keywords)",
                        dest='bookmark_query', action='store_true',
                        default=False,required=False)
    parser.add_argument('-K', help="Import keywords (no search)",
                        dest='keyword_query', action='store_true',
                        default=False,required=False)
    parser.add_argument('bookmarks', help="Bookmarks file (html format)")
    args = parser.parse_args()
    return args


def import_netscape_bookmarks(bookmarks_file, bookmark_types, output_format):
    """Import bookmarks from a NETSCAPE-Bookmark-file v1.

    Generated by Chromium, Firefox, IE and possibly more browsers
    """
    import bs4
    with open(bookmarks_file, encoding='utf-8') as f:
        soup = bs4.BeautifulSoup(f, 'html.parser')
    bookmark_query = {
        'search':
        lambda tag: (tag.name == 'a') and ('shortcuturl' in tag.attrs) and ('%s' in tag['href']),
        'keyword':
        lambda tag: (tag.name == 'a') and ('shortcuturl' in tag.attrs) and ('%s' not in tag['href']),
        'bookmark':
        lambda tag: (tag.name == 'a') and ('shortcuturl' not in tag.attrs) and (tag.string)
    }
    output_template = {
        'ncsearch': {
            'search': "config.val.url.searchengines['{tag[shortcuturl]}'] = '{tag[href]}' #{tag.string}"
        },
        'search': {
            'search': '{tag[shortcuturl]} = {tag[href]} #{tag.string}',
        },
        'bookmark': {
            'bookmark': '{tag[href]} {tag.string}',
            'keyword': '{tag[href]} {tag.string}'
        },
        'quickmark': {
            'bookmark': '{tag.string} {tag[href]}',
            'keyword': '{tag[shortcuturl]} {tag[href]}'
        }
    }
    bookmarks = []
    for typ in bookmark_types:
        tags = soup.findAll(bookmark_query[typ])
        for tag in tags:
            if typ=='search':
                tag['href'] = tag['href'].replace('%s','{}')
            if tag['href'] not in bookmarks:
                bookmarks.append(output_template[output_format][typ].format(tag=tag))
    for bookmark in bookmarks:
        print(bookmark)


if __name__ == '__main__':
    main()
