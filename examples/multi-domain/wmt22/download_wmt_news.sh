LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

if [ ! -d $DATA_DIR/wmtnews ]; then
  mkdir $DATA_DIR/wmtnews
fi

for year in 2008 {2010..2021};do
  echo "Downloading Chinese wmtnews data for year $year..."
  wget --no-check-certificate -q -O "$DATA_DIR/wmtnews/zh.${year}.gz" "https://data.statmt.org/news-crawl/zh/news.${year}.zh.shuffled.deduped.gz" \
    && gunzip "$DATA_DIR/wmtnews/zh.${year}.gz" &
  pwait 10
done

for year in {2007..2021};do
  echo "Downloading English wmtnews data for year $year..."
  wget --no-check-certificate -q -O "$DATA_DIR/wmtnews/en.${year}.gz" "https://data.statmt.org/news-crawl/en/news.${year}.en.shuffled.deduped.gz" \
    && gunzip "$DATA_DIR/wmtnews/en.${year}.gz" &
  pwait 10
done
wait

cat $DATA_DIR/wmtnews/zh.* > $DATA_DIR/wmtnews/zh.news
cat $DATA_DIR/wmtnews/en.* > $DATA_DIR/wmtnews/en.news
