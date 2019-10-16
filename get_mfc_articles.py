from nexis_driver import NexisDriver

driver = NexisDriver(download_dir_name="downloads",
                     chrome_driver_path="./media_frames_corpus/chromedriver")

driver.get_query_article_urls("A Set of Borders to Cross; For Children Seeking Immigrant Relatives in U.S., Journey Is Twofold")

