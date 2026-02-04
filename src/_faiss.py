    import _faiss
    mat_trajectories = []
    print("Building matrix")
    for i in range(T - traj_length + 1):
        traj = data[i:i+traj_length].reshape(-1)
        mat_trajectories.append(traj)
    mat_trajectories = np.array(mat_trajectories).astype(np.float32)
    del data
    print("Building FAISS index")

    index = _faiss.IndexFlatL2(H * W * traj_length)
    index.add(mat_trajectories)
    print(f"FAISS index has {index.ntotal} vectors.")

    start_time = time.time()
    D, I = index.search(mat_trajectories, k=k*traj_length)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"FAISS k-NN search took {elapsed:.2f} seconds.")