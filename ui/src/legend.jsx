function Legend(props) {
  return (
    <div>
      <h1 className=''>Runtime Diff Legend</h1>
      <div className="grid grid-rows-2 grid-flow-col">
        <Slice op={{'name':'Small__Slowdown'}} classes="m-1 border-neutral-900 bg-red-300"/>
        <Slice op={{'name':'Small__Speedup'}} classes="m-1 border-neutral-900 bg-blue-300"/>
        <Slice op={{'name':'Large__Slowdown'}} classes="m-1 border-neutral-900 bg-red-600 text-white"/>
        <Slice op={{'name':'Large__Speedup'}} classes="m-1 border-neutral-900 bg-blue-600 text-white"/>
        <Slice op={{'name':'Added'}} classes="m-1 border-dashed border-red-600 bg-white"/>
        <Slice op={{'name':'Removed'}} classes="m-1 border-dashed border-blue-600 bg-white"/>
      </div>
    </div>
  )
}
