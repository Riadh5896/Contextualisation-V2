mkdir -p ~/.contextualisation/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.contextualisation/config.toml