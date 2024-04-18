"""
下载国税网验证码，保存到dataset/captcha-images
"""

import os
import re
import time
import asyncio
import base64
import hashlib

import click
import pandas as pd
from loguru import logger
from pyppeteer import launch

from src import assets_path, dataset_path


class BrowserHandle:
    """浏览器操作"""

    def __init__(self, debug=True, browser_path=None):
        self.url = "https://inv-veri.chinatax.gov.cn/"
        self.debug = debug
        self.browser = None
        self.browser_path = browser_path
        self._call_async_func(self._init)
        self._prev = "https://inv-veri.chinatax.gov.cn/images/code.png"

    @staticmethod
    def _call_async_func(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        _task = asyncio.ensure_future(func(*args, **kwargs))
        loop.run_until_complete(_task)
        return _task.result()

    def fill_basement(self, inv_code, inv_num, inv_date, inv_chk):
        return self._call_async_func(
                self._fill_basement, inv_code, inv_num, inv_date, inv_chk
        )

    def get_verify_code(self, max_wait_time=15):
        return self._call_async_func(self._get_verify_code, max_wait_time=max_wait_time)

    def close(self):
        return self._call_async_func(self._close)

    async def _init(self):
        self.browser = await launch(
                headless=(not self.debug),
                ignoreHTTPSErrors=True,
                defaultViewport=None,
                executablePath=self.browser_path,
                viewport={"width": 1920, "height": 1080},
                args=[
                        "--disable-infobars",
                        "--start-maximized",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--disable-blink-features=AutomationControlled",
                ],
        )
        self.page = await self.browser.newPage()
        await self.page.evaluate(
                """() =>{ Object.defineProperties(navigator,{ webdriver:{ get: () => false } }) }"""
        )
        await self.page.goto(self.url)

    async def _fill_basement(self, inv_code, inv_num, inv_date, inv_chk):
        """填写基础信息"""
        fpdm = await self.page.waitForXPath('//*[@id="fpdm"]')
        await fpdm.click({"clickCount": 3})
        await fpdm.type(inv_code)
        fphm = await self.page.waitForXPath('//*[@id="fphm"]')
        await fphm.click({"clickCount": 3})
        await fphm.type(inv_num)
        kprq = await self.page.waitForXPath('//*[@id="kprq"]')
        await kprq.click({"clickCount": 3})
        await kprq.type(inv_date)
        kjje = await self.page.waitForXPath('//*[@id="kjje"]')
        await kjje.click({"clickCount": 3})
        await kjje.type(inv_chk)

    async def _refresh_verify_code(self):
        """refresh verify code"""
        ele_refresh = await self.page.waitForXPath('//*[@id="yzm_img"]', visible=True)
        await ele_refresh.click()

    async def _get_verify_code(self, max_wait_time=15):
        """get verify code"""
        await self._refresh_verify_code()
        cnt = 0
        curr = self._prev
        yzm_base64_str = None
        while cnt <= max_wait_time and self._prev == curr:
            cnt += 1
            time.sleep(1)
            logger.warning(f"wait verify image for {cnt}s...")
            ele_yzm = await self.page.waitForXPath('//*[@id="yzm_img"]')
            yzm_base64_str = await (await ele_yzm.getProperty("src")).jsonValue()
            if yzm_base64_str is not None:
                self._prev = curr
                curr = hashlib.md5(yzm_base64_str.encode("utf-8")).hexdigest()
        if self._prev == curr:
            logger.error("couldn't get verify image!")
            return None, None
        ele_info = await self.page.waitForXPath('//*[@id="yzminfo"]')
        tip = await (await ele_info.getProperty("textContent")).jsonValue()
        return yzm_base64_str, tip

    async def _close(self):
        if self.browser:
            await self.browser.close()

    def __del__(self):
        if self.browser:
            self._call_async_func(self.browser.close)


def tip_to_channel(tip: str):
    if "红色" in tip:
        channel = "red"
    elif "蓝色" in tip:
        channel = "blue"
    elif "黄色" in tip:
        channel = "yellow"
    elif tip == "请输入验证码文字":
        channel = "black"
    else:
        raise ValueError(f"Unknown tip message:{tip}")
    return channel


def save_base64_img(base64_str: str, save_dir: str, tip: str):
    """base64对象转换为图片"""
    channel = tip_to_channel(tip)
    file_path = f"{save_dir}/{channel}-{int(time.time() * 1000)}.png"
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"save captcha img to {file_path}")
    base64_data = re.sub("^data:image/.+;base64,", "", base64_str)
    byte_data = base64.b64decode(base64_data)
    with open(file_path, "wb") as fb:
        fb.write(byte_data)


def task(inv_data: pd.DataFrame, save_dir, num, debug=False, browser_path=None):
    brow = BrowserHandle(debug=debug, browser_path=browser_path)
    cnt = 0
    while cnt < num:
        row_data = inv_data.sample(n=1).iloc[0]
        brow.fill_basement(
                row_data.InvCode, row_data.InvNum, row_data.InvDate, row_data.InvCheckCode
        )
        while cnt < num:
            cnt += 1
            logger.info(
                    f"[{cnt}/{num}] InvCode:{row_data.InvCode}, InvNum:{row_data.InvNum}, "
                    f"InvDate:{row_data.InvDate}, InvCheckCode:{row_data.InvCheckCode}"
            )
            try:
                base64_img, tip = brow.get_verify_code()
                save_base64_img(base64_img, save_dir, tip)
            except Exception as e:
                logger.error(f"error:{e}")
                break


@click.command()
@click.option("-s", "--save_path", help="output directory")
@click.option("-n", "--num", default=10000, help="how many captcha images to download")
@click.option("-d", "--debug", is_flag=True, help="debug mode")
@click.option("-b", "--browser_path", help="chrome browser path")
def main(save_path: str, num: int, debug: bool, browser_path: str):
    inv_data = pd.read_csv(str(assets_path / "inv_data.csv"), encoding="utf-8", dtype=str)
    inv_data = inv_data.reset_index(drop=True)
    if not save_path:
        save_path = str(dataset_path / "origin")
    task(inv_data, save_path, num, debug=debug, browser_path=browser_path)


if __name__ == "__main__":
    main()
