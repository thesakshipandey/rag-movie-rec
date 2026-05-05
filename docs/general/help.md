# SERVER HELP CHEATSHEET (TXT)

## TMUX

tmux new -s <name>
tmux ls
tmux attach -t <name>
tmux kill-session -t <name>

Shortcuts (prefix = Ctrl+b):
Ctrl+b %        # vertical split
Ctrl+b "        # horizontal split
Ctrl+b arrow    # move
Ctrl+b d        # detach

## DISK SPACE

du -h            # human‑readable sizes for all subdirs (many lines)
du -s <path>     # summarize total for each argument (one line)
du -sh .         # human‑readable total for current dir (one line)
du -sh \* | sort -h   # per immediate child, sorted by size

## MOVE FILES (bee.cse.iitb.ac.in)

Upload:
scp <file> [sakshipandey@bee.cse.iitb.ac.in](mailto:sakshipandey@bee.cse.iitb.ac.in):/path/

Download:
scp [sakshipandey@bee.cse.iitb.ac.in](mailto:sakshipandey@bee.cse.iitb.ac.in):/path/file .

Fast sync (resumable):
rsync -avhP <src>/ [sakshipandey@bee.cse.iitb.ac.in](mailto:sakshipandey@bee.cse.iitb.ac.in):/dst/

## PROCESSES & RESOURCES

top                      # processes (use htop if available)
free -h                  # memory
vmstat 1                 # cpu snapshot every 1s
iostat -xz 1             # disk io per device
nvidia-smi               # gpu
ps aux --sort=-%mem | head   # top memory users
pkill -f "<pattern>"         # kill by pattern

## NETWORK & PORTS

ss -ltnp                 # listening ports (pid/program)
sudo lsof -i :3000       # what uses port 3000
curl -I [http://localhost:8000](http://localhost:8000)   # quick HTTP check

## NAVIGATION (FAST)

pwd                      # where am I
ls -lah                  # list with sizes/hidden
cd -                     # jump back to previous dir
pushd <dir> / popd       # directory stack; use 'dirs -v' to see
find . -maxdepth 7 -print | sed -E 's;[^/]+/;│   ;g; s;│   ([^│]+)$;└── \1;'    # show folder tree depth 2 (install: tree)
<!-- tree -L 2 -->
find . -maxdepth 1 -type d   # only dirs here

## SEARCH FILES & INSIDE FILES

find <root> -name "\*.py" -size +1M
grep -R "<text>" <root>
rg "<text>" <root>        # ripgrep (faster, if installed)

## WHO OWNS THIS PID / WHAT IS IT?

ps -p <PID> -o pid,user,%cpu,%mem,etime,cmd
pstree -aps <PID>             # show parent/children (install: pstree)
readlink -f /proc/<PID>/exe   # path to the executable
lsof -p <PID> | head          # open files / sockets of a PID
renice -n 10 -p <PID>         # lower priority (needs perms)

## GPU — SEE WHO’S USING AND HOW MUCH

nvidia-smi                         # summary (like you ran)
nvidia-smi -i <GPU_ID>             # focus on one GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi pmon -c 1               # per‑process GPU util snapshot
nvidia-smi dmon -s pucm -d 1       # device monitor every 1s (power, util, mem)
watch -n 1 nvidia-smi              # live refresh in terminal
nvtop                               # nice TUI if installed

## MAP GPU PID → USER

* Example: PID 3183909 from nvidia-smi

ps -p 3183909 -o pid,user,tty,stime,etime,cmd

* Or list all GPU PIDs with owners

for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do&#x20;
ps -p \$p -o pid,user,cmd; done

## DIRECT (ONE‑LINERS)

* Kill whatever is on port 3000

sudo lsof -t -i :3000 | xargs -r sudo kill -9

* Find big files (>1G) under /var

sudo find /var -type f -size +1G -exec ls -lh {} ; | awk '{print \$5, \$9}'

* See which process is hammering disk right now

sudo iotop -oPa

## NOTES

• Keep this file as help.txt and open with:  less help.txt
• Copy commands by mouse selection (Shift may be needed inside tmux).
