import requests
import sys

# Function WIP
def main():
    return
    # Check for arguments
    if len(sys.argv) < 2:
        print("Usage: python3 prep-link.py <link1/file1> [link2] ...")
        sys.exit(1)
    
    # Determine if link or text file

    # Get list of links
    args = sys.argv[1:]
    for arg in args:
        print(f"Processing {arg}")

        # Fetch link and save file
        resp = requests.get(arg)
        if resp.ok:
            print (resp.text)
        else:
            print ("Failed. Status {}".format(resp.status_code))
            print (resp.text)


if __name__ == "__main__":
    main()
