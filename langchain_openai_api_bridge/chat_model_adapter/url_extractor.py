def extract_base64_url(url) -> tuple[str, str]:
    data_base64_url = url
    _, data_base64 = data_base64_url.split(";base64,")
    start_index = url.find("data:") + 5
    end_index = url.find(";", start_index)
    media_type = url[start_index:end_index]

    return (media_type, data_base64)
