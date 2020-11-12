mkdir -p ~/.streamlit/
echo "\
[general]\
email = \"jariwala@uw.edu\"\
" > ~/.streamlit/credentials.toml
echo "\
[server]\
headless = true\
enableCORS=false\
port = $PORT\"\