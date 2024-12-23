import requests

# 替换以下内容为你的用户名和密码（如果需要登录）
identity = "2820811978@qq.com"  # 替换为你的Space-Track用户名
password = "Li*****20021114"  # 替换为你的Space-Track密码
spacetrack_csrf_token = "c25f6e65874fb5ea28aff5763b5ab3cc"
beginDate = "2023-12-31"
endDate = "2024-02-02"

if __name__ == "__main__":

    # TLE 数据的 URL
    url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/41335/orderby/TLE_LINE1 ASC/EPOCH/{beginDate}--{endDate}/format/tle"
    # Space-Track 的认证信息
    login_url = "https://www.space-track.org/ajaxauth/login"
    session = requests.Session()
    # login_payload = {"spacetrack_csrf_token":spacetrack_csrf_token,"identity": identity, "password": password,"btnLogin":"LOGIN"}
    login_payload = {"identity": identity, "password": password}
    login_response = session.post(login_url, json=login_payload)

    # 检查是否登录成功
    if login_response.status_code == 200:
        print("登录成功！")
        # 获取 TLE 数据
        response = session.get(url)
        if response.status_code == 200:
            file_path = "../dataset/dataTLE.txt"
            lines = response.text.strip().split('\n')
            tle_data = []
            for line in lines:
                now_line = line.strip()
                if now_line:
                    tle_data.append(now_line)
            tle_data = "\n".join(tle_data)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(tle_data)
            print(f"TLE 数据已保存到 {file_path}")
        else:
            print(f"获取 TLE 数据失败，状态码: {response.status_code}")
            print("错误信息:", response.text)
    else:
        print(f"登录失败，状态码: {login_response.status_code}")
        print("错误信息:", login_response.text)

