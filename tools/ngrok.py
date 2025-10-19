from pyngrok import ngrok
ngrok.set_auth_token("32Pf3JDFeJTHcRoi0NkCqgUVV68_48P3gyynHQFuwqp8QJpC4")  # опционально
tunnel = ngrok.connect(8000)
print(tunnel.public_url)