# Daily Data Update Scheduling

Automates the daily data fetch to keep your trading data up-to-date.

## Option 1: macOS Launchd (Recommended for Mac)

**Advantages:**
- Native macOS scheduler
- Reliable, lightweight
- Runs even when you're not logged in
- Can wake computer to run jobs

**Setup:**

```bash
# 1. Copy the plist file to LaunchAgents
cp setup/com.quant.daily_update.plist ~/Library/LaunchAgents/

# 2. Load the job
launchctl load ~/Library/LaunchAgents/com.quant.daily_update.plist

# 3. Verify it's loaded
launchctl list | grep quant

# 4. (Optional) Test run immediately
launchctl start com.quant.daily_update
```

**Uninstall:**

```bash
# Unload the job
launchctl unload ~/Library/LaunchAgents/com.quant.daily_update.plist

# Remove the file
rm ~/Library/LaunchAgents/com.quant.daily_update.plist
```

**Logs:**

- Scheduler output: `logs/daily_update_stdout.log`
- Errors: `logs/daily_update_stderr.log`
- Application logs: `logs/daily_updates_YYYY-MM-DD.log`

---

## Option 2: Cron (Simple, cross-platform)

**Advantages:**
- Works on Mac, Linux, WSL
- Very simple
- Standard Unix tool

**Setup:**

```bash
# 1. Open crontab editor
crontab -e

# 2. Add this line (runs at 6 PM every weekday)
0 18 * * 1-5 cd /Users/jwassink/repos/quant_trade && .venv/bin/python scripts/update_daily_data.py >> logs/cron.log 2>&1

# 3. Save and exit (press 'i' to insert, ESC then ':wq' to save in vi)

# 4. Verify it's installed
crontab -l
```

**Cron Schedule Format:**
```
 ┌─── minute (0-59)
 │ ┌─── hour (0-23)
 │ │ ┌─── day of month (1-31)
 │ │ │ ┌─── month (1-12)
 │ │ │ │ ┌─── day of week (0-7, Sunday=0 or 7)
 │ │ │ │ │
 * * * * * command
```

**Common Schedules:**

```bash
# Every day at 6 PM
0 18 * * * <command>

# Monday-Friday at 6 PM (after market close)
0 18 * * 1-5 <command>

# Every Sunday at 8 AM (weekly updates)
0 8 * * 0 <command>

# Every hour during trading hours (9 AM - 4 PM, weekdays)
0 9-16 * * 1-5 <command>
```

---

## Option 3: Python APScheduler (In-process scheduler)

**Advantages:**
- Pure Python
- Flexible scheduling
- Good for development/testing

**Setup:**

```bash
# 1. Install APScheduler
.venv/bin/pip install apscheduler

# 2. Create a scheduler script (see scripts/run_scheduler.py)
# 3. Run it in the background
nohup .venv/bin/python scripts/run_scheduler.py &
```

**Not implemented yet - let me know if you want this option!**

---

## Recommended Schedule

**Daily (Weekdays):**
- **6:00 PM ET**: Run price data update
  - US markets close at 4 PM ET
  - 2-hour buffer for data to be available

**Weekly (Sundays):**
- **8:00 AM**: Update metadata and financials
  - Low traffic time
  - Ready for Monday

---

## Monitoring

**Check if scheduler is running:**

```bash
# Launchd
launchctl list | grep quant

# Cron
crontab -l
ps aux | grep update_daily_data
```

**View logs:**

```bash
# Recent updates
tail -f logs/daily_updates_*.log

# Last run
ls -lt logs/daily_updates_*.log | head -1

# Errors
grep ERROR logs/daily_updates_*.log
```

**Manual run (for testing):**

```bash
# Run immediately
.venv/bin/python scripts/update_daily_data.py

# Force metadata/financials update (even if not Sunday)
# (Edit update_daily_data.py and set force=True in main())
```

---

## Troubleshooting

**Job doesn't run:**

1. Check if scheduler is loaded: `launchctl list | grep quant`
2. Check paths are absolute (not relative) in plist file
3. Check Python virtual environment path is correct
4. Review error logs

**Missing data:**

1. Check if today is a trading day (not weekend/holiday)
2. Check network connection
3. Check yfinance API is working: `python -c "import yfinance; print(yfinance.__version__)"`
4. Review application logs for errors

**Performance issues:**

1. Reduce universe size for testing
2. Increase `lookback_days` if missing data
3. Add retry logic for failed symbols

---

## Next Steps

1. **Choose your scheduler** (recommend Launchd for Mac, Cron for Linux)
2. **Test manually** first: `.venv/bin/python scripts/update_daily_data.py`
3. **Install scheduler** following instructions above
4. **Monitor for a week** to ensure it's working
5. **Set up alerts** (optional - email/Slack on failures)

## Future Enhancements

- [ ] Add US market holiday calendar (pandas_market_calendars)
- [ ] Email/Slack notifications on failures
- [ ] Retry failed symbol fetches
- [ ] Data quality checks (detect missing/corrupt data)
- [ ] Performance metrics (track fetch time)
- [ ] Backfill gaps automatically
