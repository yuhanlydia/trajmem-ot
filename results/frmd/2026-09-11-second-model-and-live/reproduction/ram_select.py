"""Select numeric-first episodes from a verified HDF5 tar archive using temporary RAM."""
import argparse, hashlib, json, os, pathlib, shutil, tarfile
import h5py


def select(archive, output, expected_sha, count=10):
    archive, output = pathlib.Path(archive), pathlib.Path(output)
    if output.exists():
        raise FileExistsError(output)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError('Archive SHA-256 mismatch')
    fd = os.memfd_create('trajmem-raw-h5', flags=os.MFD_CLOEXEC)
    partial = output.with_suffix('.h5.partial')
    try:
        with tarfile.open(archive, mode='r|xz', bufsize=1024*1024) as tar:
            member = tar.next()
            if not member or not member.isfile() or not member.name.endswith('.h5'):
                raise ValueError('Expected a single HDF5 archive member')
            stat = dict(line.split() for line in pathlib.Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            limit = int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
            nonreclaimable = int(stat.get('anon', 0)) + int(stat.get('shmem', 0)) + int(stat.get('kernel', 0))
            if member.size + nonreclaimable + 8*1024**3 >= limit:
                raise MemoryError('Insufficient container memory with 8 GiB reserve')
            print(f'Decompressing {member.name}: {member.size} bytes into temporary RAM', flush=True)
            source = tar.extractfile(member)
            with os.fdopen(os.dup(fd), 'wb') as destination:
                shutil.copyfileobj(source, destination, length=8*1024*1024)
            # Consume the remainder so XZ stream integrity is checked.
            if tar.next() is not None:
                raise ValueError('Expected exactly one HDF5 member')
        if os.fstat(fd).st_size != member.size:
            raise ValueError('Decompressed size mismatch')
        output.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(os.dup(fd), 'rb') as memory_file, h5py.File(memory_file, 'r') as source, h5py.File(partial, 'w') as dest:
            episodes = sorted((key for key in source if key.startswith('episode_')), key=lambda k: int(k.split('_')[1]))
            if len(episodes) < count:
                raise ValueError('Source has fewer episodes than requested')
            selected = episodes[:count]
            for key in selected:
                source.copy(key, dest)
                print('Copied', key, flush=True)
            for key, value in source.attrs.items():
                dest.attrs[key] = value
        with h5py.File(partial, 'r') as dest:
            if set(dest.keys()) != set(selected):
                raise ValueError('Selected episode membership mismatch')
        partial.rename(output)
        report = {'archive': archive.name, 'archive_sha256': digest.hexdigest(), 'source_h5_bytes': member.size,
                  'source_episode_count': len(episodes), 'selected_episodes': selected,
                  'selection_rule': 'First numeric episode IDs, independent of actions or outcomes',
                  'selected_h5_bytes': output.stat().st_size, 'temporary_storage': 'memfd, released after selection'}
        output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    finally:
        os.close(fd)
        partial.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--count', type=int, default=10)
    a = p.parse_args()
    print(json.dumps(select(a.archive, a.output, a.sha256, a.count), indent=2))
