
import React, {useState, useEffect, useLayoutEffect} from 'react';
import { useSelector,  useDispatch } from 'react-redux';
import { setSearch } from './redux/searchSlice';
import { setWidth, setModel, setSummarizeResources, setColorize, setDarkMode, selectClick, setResourceFilter, setShowOptions, setShowWorkloadsTable, addWorkloadOption } from './redux/appSlice';
import { displayNextLevel } from './redux/levelsSlice';
import Level from './level';
import 'bootstrap-icons/font/bootstrap-icons.css';
import ReactVirtualizedTable from './table';
import SyntaxHighlight from "./syntaxHighlight";

import theme from "prism-react-renderer/themes/vsLight"; // Themes: dracula  duotoneDark  duotoneLight  github  nightOwl  nightOwlLight  oceanicNext  okaidia  palenight  shadesOfPurple  synthwave84  ultramin  vsDark  vsLight.
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Checkbox from '@mui/material/Checkbox';
import Paper from '@mui/material/Paper';
import { styled } from '@mui/material/styles';
import clsx from "clsx";
import { FormControlUnstyledContext } from '@mui/base';
import { Helmet } from "react-helmet";


import './index.css';


var loadMode = 'NOT github';
const githubBaseUrl = 'https://raw.githubusercontent.com/danielsnider/hotline-results/main/3090-x1-gpu/';
const githubResultsList = [
  'src/results/RNN.json',
  'src/results/GNN.json',
  'src/results/DLRM.json',
  'src/results/ResNet50-batch-16.json',
  'src/results/ResNet50.json',
  'src/results/Stable Diffusion (Inference).json',
  'src/results/Transformer.json',
]

let loadDomain = null;
if (loadMode === 'github') {
  loadDomain = githubBaseUrl + 'dist/';
} else {
  const currentUrl = window.location.href;
  loadDomain = currentUrl.split('/').slice(0, 3).join('/') + '/'; // get the current domain and port such as "http://localhost:7234/"
}


// Docs on dependency resolution: https://parceljs.org/features/dependency-resolution/#glob-specifiers
// const results = import('./example-results/jan2023/*.js'); // example results to be displayed in UI
// const results = import('./results/*.js');  // user results to be displayed in UI
// const results = {...user_results, ...example_results};  # doesn't load fast enough... need to use redux and Promise.all then trigger a re-render in App() useEffect()


const Mousetrap = require("mousetrap");


function Options(props) {
  const dispatch = useDispatch();

  const summarizeResources = useSelector((state) => state.view.summarizeResources);
  function toggleSummarizeResources(){
    dispatch(setSummarizeResources(!summarizeResources));
  }

  const colorize = useSelector((state) => state.view.colorize);
  function toggleColorize(){
    dispatch(setColorize(!colorize));
  }

  const darkMode = useSelector((state) => state.view.darkMode);
  function toggleDarkMode(){
    const htmlEl = document.querySelector('html');
    if (!darkMode) {
      localStorage.theme = 'dark';
      htmlEl.classList.add('dark');
    } else {
      localStorage.theme = 'light';
      htmlEl.classList.remove('dark');
    }
    dispatch(setDarkMode(!darkMode));
  }

  const resourceFilterOptions = [
    'All',
    'CPU',
    'GPU'
  ]

  function resourceFilterHandler(e){
    const resourceFilter = e.target.value;
    dispatch(setResourceFilter(resourceFilter));
  }

  return (
    <div>
      <div className='flex pb-1 pl-1'>
        <Search />
        <span className='max-h-6 break-all overflow-hidden'>
          Display only longest resource per type:
        </span>
        <form className='pl-1 pt-[0.5]'>
          <input
            type="checkbox"
            checked={summarizeResources}
            onChange={toggleSummarizeResources}
          />
        </form>
        <span className='pl-4 max-h-6 break-all overflow-hidden'>Colorize:</span>
        <form className='pl-1 pt-[0.5]'>
          <input
            type="checkbox"
            checked={colorize}
            onChange={toggleColorize}
          />
        </form>
        {/* <span className='pl-4 max-h-6 break-all overflow-hidden'>Dark Mode:</span>
        <form className='pl-1 pt-[0.5]'>
          <input
            type="checkbox"
            checked={darkMode}
            onChange={toggleDarkMode}
          />
        </form> */}
        <span className='pl-4 max-h-6 break-all overflow-hidden'>Resources:</span>
        <span className='pl-2 tracking-wide drop-shadow'>
          <select
            onChange={resourceFilterHandler}
            defaultValue={'All'}
            className="browser-default custom-select" >
            {
              resourceFilterOptions.map((val, key) => <option key={key} value={val}>{val}</option>)
            }
          </select>
        </span>
      </div>
    </div>
  )
}


const WorkloadsTable = (props) => {
  const dispatch = useDispatch();
  const [selectedRow, setSelectedRow] = useState(0);
  const model = useSelector((state) => state.model.value)
  if (!model) {
    return
  }
  const selectedModelId = model.id;

  const handleModelChange = (row) => {
    setSelectedRow(row.id);
    props.toggleWorkloadsTable();
    dispatch(setModel(row.model));
    dispatch(selectClick({op: row.model}));
  };

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    "&.Mui-selected": {
      backgroundColor: "#dadada",
      "&:hover": {
        backgroundColor: "#dadada",
      },
    },
    "&:hover:not(.MuiTableRow-head):not(.Mui-selected)": {
      backgroundColor: "#f1f1f1",
      cursor: 'pointer',
    },
  }));

  const StyledTableCell = styled(TableCell)({
    "&.MuiTableCell-head": {
      fontWeight: "bold",
      background: "#f3f3f3",
    },
    "&.MuiTableCell-root": {
      padding: "4px",
    },
  });

  const workloadOptions = useSelector((state) => state.view.workloadOptions);

  return (
    <div className='pb-4' style={props.style}>
      <Paper variant="outlined" >
        <Table>
          <TableHead>
            <StyledTableRow>
              <StyledTableCell></StyledTableCell>
              <StyledTableCell>Model</StyledTableCell>
              <StyledTableCell>Dataset</StyledTableCell>
              <StyledTableCell>Optimizer</StyledTableCell>
              <StyledTableCell>Batch</StyledTableCell>
              <StyledTableCell>GPU</StyledTableCell>
              <StyledTableCell>Runtime</StyledTableCell>
            </StyledTableRow>
          </TableHead>
          <TableBody>
            {
            workloadOptions.map((row) => (
              <StyledTableRow
                key={row.id}
                selected={row.id === selectedModelId}
                onClick={() => handleModelChange(row)}
              >
                <StyledTableCell>
                  <Checkbox
                    checked={row.id === selectedModelId}
                    color="default"
                    sx={{ '& .MuiSvgIcon-root': { fontSize: 22 } }}
                  />
                </StyledTableCell>
                <StyledTableCell>{row.model_name}</StyledTableCell>
                <StyledTableCell>{row.dataset}</StyledTableCell>
                <StyledTableCell>{row.optimizer}</StyledTableCell>
                <StyledTableCell>{row.batch}</StyledTableCell>
                <StyledTableCell>{row.gpu.replace('GeForce RTX','')}</StyledTableCell>
                <StyledTableCell>{row.runtime}</StyledTableCell>
              </StyledTableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </div>
  );
};


function WorkloadCard(props) {
  if (!props.text) {
    return
  }
  return (
    <div className="mt-2 mr-2">
    <div id={'card-'+props.title} className="rounded-lg border border-gray-300 bg-white px-3 py-2 cursor-pointer hover:bg-gray-200">
      <div className="text-gray-500 text-sm font-medium">{props.title}</div>
      <div className="text-black text-xl font-bold" style={{maxHeight: '56px'}}>{props.text}</div>
    </div>
  </div>

  )
}

function WorkloadCards(props) {
  const model = useSelector((state) => state.model.value);
  if (!model) {
    return
  }
  return (
    <div className='flex '>
      <div className="flex flex-wrap overflow-hidden" style={{height: '81px'}}>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='Model' text={model["metadata.model"]}/>
        </div>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='Dataset' text={model["metadata.dataset"]}/>
        </div>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='Optimizer' text={model["metadata.optimizer"]}/>
        </div>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='Batch' text={model["metadata.batch_size"]}/>
        </div>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='GPU' text={model["gpu_model"]}/>
        </div>
        <div className="grow" onClick={props.toggleWorkloadsTable}>
          <WorkloadCard title='Runtime' text={model["runtime_str"] }/>
        </div>
      </div>
      <div className=" flex text-center" onClick={props.toggleShowOptions}>
        <WorkloadCard title='Options' text={
          <i className="bi bi-gear-fill"></i>
        } />
      </div>
    </div>
  )
}

export function Logo(props) {
  return (
    <div className={props.containerClass}>
      <div id='logo' style={{
        fontFamily: "'ngc-gradient'",
        fontSize: '56px',
        lineHeight: '40px',
        backgroundColor: '#f3ec78',
        backgroundImage: 'linear-gradient(45deg, #f3ec78, #af4261)', // hot colors
        backgroundSize: '100%',
        paddingRight: '0.2em',
        marginRight: '-0.2em',
        position: 'absolute',
        top: '8px',
        left: '0px',
        zIndex: '1',

        }}>
          HOTLINE
          </div>
      <div  style={{
        fontFamily: "'ngc-gradient'",
        fontSize: '56px',
        lineHeight: '40px',
        color: 'black',
        paddingRight: '0.2em',
        marginRight: '-0.2em',
        zIndex: '2',
        }}>
          HOTLINE
      </div>
      </div>
  )
}


function Header(props) {
  const dispatch = useDispatch();

  const showOptions = useSelector((state) => state.view.showOptions);
  function toggleShowOptions(){
    dispatch(setShowOptions(!showOptions));
    var elem = document.getElementById('level-1-track-0');
    if (elem) {
      // Trigger re-render of hotline timeline since the height above it changed.
      var smallChange = showOptions ? 0.00001 : -0.00001;
      dispatch(setWidth(elem.clientWidth + smallChange));
    }
    dispatch(setShowOptions(!showOptions));
  }

  const showWorkloadsTable = useSelector((state) => state.view.showWorkloadsTable);
  function toggleWorkloadsTable(){
    var elem = document.getElementById('level-1-track-0');
    if (elem) {
      // Trigger re-render of hotline timeline since the height above it changed.
      var smallChange = showWorkloadsTable ? 0.00002 : -0.00002;
      dispatch(setWidth(elem.clientWidth + smallChange));
    }
    dispatch(setShowWorkloadsTable(!showWorkloadsTable));
  }

  return (
    <div id='header' className='px-2 pt-2'>
      <div className="flex w-full">
        <div className="w-80 pr-1 pb-1">
          <div className='text-center'>
            <Logo containerClass='p-1 h-[40px] relative'/>
            <div className='pt-[8px] text-stone-500'
              style={{
                fontFamily: "'SF1', sans-serif",
                fontSize: '14px' }}>
                PROFILE TIME-USE IN DNNS
            </div>
          </div>
        </div>
        <div className="w-full pb-1 pl-4">
          <div className="flex flex-col">
            <div className="flex flex-col md:flex-row flex-col-reverse">
            </div>
            <div className="h-full">
              <WorkloadCards toggleWorkloadsTable={toggleWorkloadsTable} toggleShowOptions={toggleShowOptions} />
            </div>
          </div>
        </div>
      </div>
      {showWorkloadsTable ? <WorkloadsTable toggleWorkloadsTable={toggleWorkloadsTable} /> : null } {/* NOTE: Trying to animate grow down workload table is BUGGED because leader lines are not in container and too fussy to get working */}
      {showOptions ? <Options/> : null }
    </div>
  )
}

function Search(props) {
  const dispatch = useDispatch()

  function debounce(func, wait, immediate) {
    var timeout;
    return function executedFunction() {
      var context = this;
      var args = arguments;
      var later = function() {
        timeout = null;
        if (!immediate) func.apply(context, args);
      };
      var callNow = immediate && !timeout;
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
      if (callNow) func.apply(context, args);
    };
  };
  var onChange = debounce(function(value) {
    dispatch(setSearch(value));
  }, 160);

  return (
    <div>
      <input
        type='text'
        id='main-search'
        className='w-80 py-1 px-2 mr-[17px] rounded-lg border border-gray-300'
        style={{transform:'translate(0px, -5px)'}}
        onChange={e => onChange(e.target.value)}
        placeholder='Search...'
      />
    </div>
  );
}

function Viewer(props) {
  const levels = useSelector((state) => state.levels.value)
  if (!levels) {
    return
  }
  const levelComponents = [];
  Object.keys(levels).forEach((key) => {
    levelComponents.push(<Level key={key} op={levels[key].op} ops={levels[key].ops} levelIdx={key} name={levels[key].name} />)
  });

  // // Draw leader for left help panel
  // useLayoutEffect(() => {
  //   if (true){
  //       const start = LeaderLine.pointAnchor({element: document.getElementById('card-Model'), x:'50%', y: '100%'});
  //       const end = LeaderLine.pointAnchor({element: document.getElementById('left-help-panel'), x:'100%', y: '50%'});
  //       const line = new LeaderLine({
  //           start: start,
  //           end: end,
  //           path: 'grid',
  //           startPlug: 'behind',
  //           endPlug: 'square',
  //           size: 1,
  //           color: 'grey',
  //       }
  //       );
  //     return function cleanup() {
  //       line.remove();
  //     }
  //   }
  // }, []);


  return (
    <div id='timeline-viewer' className="mx-2 overflow-x-hidden bg-white">
      <div className='border-[3px] rounded border-neutral-900'>
        {levelComponents}
      </div>
    </div>
  );
}

function Shortcuts(props) {
  const selectedSlice = useSelector((state) => state.select.value);
  const model = useSelector((state) => state.model.value)
  function keyPressed(event) {
    // console.log(event);
    if (!event) {
      return
    }
    if (['ArrowRight', 'ArrowLeft'].includes(event.key) && !(selectedSlice && 'id' in selectedSlice)) {
      // If nothing is selected, select the first available op.
      defaultResource = Object.keys(model.ops[0].resources)[0];
      defaultSliceId = model.ops[0].id;
      defaultElemId = `${defaultResource}-${defaultSliceId}`;
      const selectedElem = document.getElementById(defaultElemId);
      if (selectedElem) {
        selectedElem.click();
      }
      return
    }
    else if (event.key == 'ArrowRight') {
      const sliceElemId = `${selectedSlice.resourceName}-${selectedSlice.id}`
      const selectedElem = document.getElementById(sliceElemId);
      const nextSlice = selectedElem.nextSibling;
      if (nextSlice) {
        nextSlice.click();
      }
    }
    else if (event.key == 'ArrowLeft') {
      const sliceElemId = `${selectedSlice.resourceName}-${selectedSlice.id}`
      const selectedElem = document.getElementById(sliceElemId);
      const nextSlice = selectedElem.previousSibling;
      if (nextSlice) {
        nextSlice.click();
      }
    }
    else if (event.key == 'ArrowUp' && event.shiftKey) {
      // Going up a level is not yet implemented
    }
    else if (event.key == 'ArrowDown' && event.shiftKey) {
      // Going down a level is not yet implemented
    }
    else if (event.key == '/' && !event.shiftKey) {
      const searchElem = document.getElementById('main-search');
      if (searchElem) {
        searchElem.focus();
        searchElem.select();
        event.preventDefault();
        return false;
      }
    }
    else if (event.key == '?') {  // this key is "shift+/"
      const searchElem = document.getElementById('slice_search_input');
      if (searchElem) {
        searchElem.focus();
        searchElem.select();
        event.preventDefault();
        return false;
      }
    }
  }

  Mousetrap.bind(["left", "right", "shift+up", "shift+down", "/", "shift+/"], keyPressed);
  useEffect(() => {
    return () => {
      Mousetrap.bind(["left", "right", "shift+up", "shift+down", "/", "shift+/"], keyPressed);
    };
  }, []);

  // React.useLayoutEffect(keyPressed,[]); // Click first slice immediately test the results. NOTE: For dev testing only.

  return (
    <>
    </>
  );
}

function Listeners(props) {
  const dispatch = useDispatch();

  function onResize() {
    var elem = document.getElementById('level-1-track-0');
    if (elem) {
      dispatch(setWidth(elem.clientWidth));
    }
  }
  React.useLayoutEffect(onResize,[]); // Call once on first render

  // Listen to window resize events with small debounce
  function debounce(func){
    return function(event){
      var timer;
      if(timer) clearTimeout(timer);
      timer = setTimeout(func,250,event);
    };
  }
  useEffect(() => {
    window.addEventListener("resize", debounce(function(e){
      onResize()
    }));
  }, []); // Call once on first render

  return (<></>);
}

function OpenWithPerfetto(props) {
  const perfettoUrl = 'https://ui.perfetto.dev';

  async function fetchAndOpen(traceUrl, traceName) {
    const resp = await fetch(traceUrl);
    const blob = await resp.blob();
    const arrayBuffer = await blob.arrayBuffer();
    openTrace(arrayBuffer, traceUrl, traceName);
  }

  function openTrace(arrayBuffer, traceUrl, traceName) {
    const win = window.open(perfettoUrl);
    if (!win) {
      console.log('Popups blocked, click here to open the trace file');
      return;
    }

    const timer = setInterval(() => win.postMessage('PING', perfettoUrl), 50);

    const onMessageHandler = (evt) => {
      if (evt.data !== 'PONG') return;

      // We got a PONG, the UI is ready.
      window.clearInterval(timer);
      window.removeEventListener('message', onMessageHandler);

      const reopenUrl = new URL(location.href);
      reopenUrl.hash = `#reopen=${traceUrl}`;
      const obj = {
        perfetto: {
          buffer: arrayBuffer,
          title: traceName,
          url: reopenUrl.toString(),
      }};
      win.postMessage(obj, perfettoUrl);
    };

    window.addEventListener('message', onMessageHandler);
  }

  const onClick = () => {
      const traceName = `${props.op.pretty_name}`;
      const traceUrl = `${loadDomain}traces${props.op.trace_file}`;
      console.log(traceUrl);
      fetchAndOpen(traceUrl, traceName);
  }

  return (
    <a
      className="py-1 px-2 rounded inline-flex items-center bg-gray-200 hover:bg-gray-300 text-gray-800 active:ring active:ring-gray-400 cursor-pointer"
      onClick={onClick}
    >
      <span>Open trace with Perfetto</span>
      <i className="bi-box-arrow-up-right pl-2"></i>
    </a>
  )
}


function OpenWithVSCode(props) {
  const isClientWindowsOS = true;
  const windowsOsPrefix = isClientWindowsOS ? '/c:' : '';
  const fileNumber = props.op.source_file_num ? ':' + props.op.source_file_num : '';
  const vscodeLink = 'vscode://file' + windowsOsPrefix + props.op.source_file_name + fileNumber;
  return (
    <div className='flex items-center justify-center'>
      <a
        className="opacity-90 ml-2 py-1 px-2 rounded inline-flex items-center bg-gray-200 hover:bg-gray-300 text-gray-800 active:ring active:ring-gray-400 cursor-pointer"
        target="_blank"
        href={vscodeLink}
      >
        <span>Open code with VSCode</span>
        <i className="bi-box-arrow-up-right pl-2"></i>
      </a>
    </div>
  )
}

function OpenWithTensorboard(props) {
  const vscodeLink = 'vscode://file' + props.op.source_file_name;
  return (
    <div className='flex items-center justify-center'>
      <a
        className="opacity-70 ml-2 py-1 px-2 rounded inline-flex items-center bg-gray-200 text-gray-800"
        target="_blank"
        // href={vscodeLink}
        style={{
          cursor: 'not-allowed',
        }}
      >
        <span>Open with Tensoboard</span>
        <i className="bi-box-arrow-up-right pl-2"></i>
      </a>
    </div>
  )
}

function OpenWithNsight(props) {
  const vscodeLink = 'vscode://file' + props.op.source_file_name;
  return (
    <div className='flex items-center justify-center'>
      <a
        className="opacity-70 ml-2 py-1 px-2 rounded inline-flex items-center bg-gray-200 text-gray-800"
        target="_blank"
        // href={vscodeLink}
        style={{
          cursor: 'not-allowed',
        }}
      >
        <span>Open with Nsight Compute</span>
        <i className="bi-box-arrow-up-right pl-2"></i>
      </a>
    </div>
  )
}


function flattenObject(ob) {
  var toReturn = {};

  for (var i in ob) {
      if (!ob.hasOwnProperty(i)) continue;

      if ((typeof ob[i]) == 'object' && ob[i] !== null) {
          var flatObject = flattenObject(ob[i]);
          for (var x in flatObject) {
              if (!flatObject.hasOwnProperty(x)) continue;

              toReturn[i + '.' + x] = flatObject[x];
          }
      } else {
          toReturn[i] = ob[i];
      }
  }
  return toReturn;
}

function Links(props) {
  return (
    <div className='flex p-8 items-center' >
      <div className="text-gray-400 mx-auto" >
        <a target="_blank" href="https://danielsnider.ca/hotline" style={{ color: 'inherit'}} className='hover:opacity-70'>About Hotline</a>
        <span className='mx-1 px-2'></span>
        <a target="_blank" href="https://danielsnider.ca/papers/Hotline.pdf" style={{ color: 'inherit'}} className='hover:opacity-70'>Paper</a>
        <span className='mx-1 px-2'></span>
        <a target="_blank" href="https://github.com/UofT-EcoSystem/hotline" style={{ color: 'inherit'}} className='hover:opacity-70'>GitHub</a>
      </div>
    </div>
  )
}


function DetailCard(props) {
  return (
    <div>
      <div className='text-3xl'>{props.title}</div>
      <div className='text-4xl font-bold'>{props.value}</div>
      <div className='py-2'>{props.subtext}</div>
      {props.button}
    </div>
  )
}

function Details(props) {
  var select_ = useSelector((state) => state.select.value)
  'use strict';

  if (!select_ || select_.id === '') {
    return
  }

  var op = {...select_, 'resources': {}};
  var slices = [];

  slices = [
      {
      'id': 'id1',
      'name': 'name1',
      'ts': 'ts1',
      'dur': 'dur1',
      'depth': 'depth1',
      'track_id': 'track_id1',
      'pid': 'pid1',
      'tid': 'tid1',
    },
    {
      'id': 'id2',
      'name': 'name2',
      'ts': 'ts2',
      'dur': 'dur2',
      'depth': 'depth2',
      'track_id': 'track_id2',
      'pid': 'pid2',
      'tid': 'tid2',
    }
  ];

  delete op.ops  // DELETE SUB OPS FROM DETAILS PANEL
  for (const resourceName of Object.keys(select_.resources)) {
    op.resources[resourceName] = {...select_.resources[resourceName]};
    if ('slices' in op.resources[resourceName]) {
      slices.push(...op.resources[resourceName].slices);
      delete op.resources[resourceName].slices; // DELETE SLICES FROM DETAILS PANEL
    }
  }
  op.slice_count = slices.length;

  return (
    <div className="pt-2 px-2">
        { op &&
          <div className="p-2 bg-neutral-50 rounded overflow-hidden">
            <div className='flex'>
              <div className='text-stone-500 font-extrabold text-xl tracking-tight'>
                Details:
              </div>
              <div className='grow pl-2 text-stone-500 text-xl'>
                {op['pretty_name']}
              </div>
              {/* <OpenWithPerfetto op={op}/> */}
            </div>
            <div className='opacity-90 pt-4 grid grid-cols-3 grid-flow-row gap-4 justify-center text-center content-center align-center text-stone-500'>
              <DetailCard
                title={'Runtime'}
                value={op.runtime_str}
                subtext={'Start time: ' + op.start_timestamp}
                // button={op.trace_file. && <OpenWithTensorboard op={op}/>}
              />
              <DetailCard
                title={'Operations'}
                value={Number(op.trace_event_count).toLocaleString()}
                subtext={'Trace size: ' + op.trace_disk_size}
                button={op.trace_file && <OpenWithPerfetto op={op}/>}
              />
              <DetailCard
                title={'Insights'}
                value={op.bound_by}
                // subtext={'Recommendations: ' + (op.recommendations || 'None')}
                // button={op.trace_file && <OpenWithNsight op={op}/>}
              />
            </div>
            { op.source_file_name &&
              <div>
                <div className='pt-4 text-stone-500 font-extrabold text-xl tracking-tight'>Code:</div>
                <div className='flex'>
                  <div className='grow'>
                    <SourceCodeLine op={op}/>
                  </div>
                  <OpenWithVSCode op={op}/>
                </div>
                <SourceCodeDisplay op={op}/>
              </div>
            }
            { op.source_stack_trace &&
              <div>
                <div className='pt-4 text-stone-500 font-extrabold text-xl tracking-tight'>Traceback:</div>
                <Traceback op={op} />
              </div>
            }
            <div className='pt-4 text-stone-500 font-extrabold text-xl tracking-tight'>Metadata:</div>
            <MetadataTable op={op} />
            {/* { slices.length > 0 && <ReactVirtualizedTable rows={slices}/> } */}
          </div>
        }
    </div>
  );
}

function SourceCodeLine(props) {
  return (
    <div className='opacity-80'>
      <div className='my-2 p-1 border-2 rounded border-neutral-300'>
        {props.op.source_file_name}:{props.op.source_file_num}
      </div>
    </div>
  )
}

function Traceback(props) {
  if (!props.op.source_stack_trace) {
    return
  }

  function scrollToBottom(){
    const scrollBox = document.getElementById('traceback-display');
    if (scrollBox) {
      scrollBox.scrollTop = scrollBox.scrollHeight;
    }
  }
  const timer = setTimeout(scrollToBottom,25);

  // Count number of lines to highlight the last line
  const lastLineNumber = (props.op.source_stack_trace.match(/\n/g) || []).length;

  return (
    <div className='opacity-80'>
      <div id="traceback-display" className='mt-2 whitespace-pre-line border-solid border-2 rounded border-neutral-300'
        style={{
          maxHeight: '200px',
          overflowY: 'scroll',
        }}
      >
        <SyntaxHighlight theme={theme} code={props.op.source_stack_trace} highlightLine={lastLineNumber} language="python" />
      </div>
    </div>
  )
}

function SourceCodeDisplay(props) {
  let sourceUrl = loadDomain + props.op.ui_source_code_path;
  if (loadMode === 'github') {
    sourceUrl = sourceUrl.replace('/home/ubuntu/cpath/results/ui/dist/', ''); // For github/hotline-results
  } else {
    sourceUrl = sourceUrl.replace('./ui/dist/', ''); // For eco machines
    // sourceUrl = sourceUrl.replace('/home/ubuntu/cpath/results/ui/dist/', ''); // For yu 3090
  }

  // console.log(sourceUrl);

  const model = useSelector((state) => state.model.value)

  const [code, setCode] = useState('');
  // Fetch source code once
  useEffect(() => {
    fetch(sourceUrl)
    .then(response => response.text())
    .then((fetchedCode) => {
      if (!fetchedCode.includes('<!DOCTYPE html>') && !fetchedCode.includes('404: Not Found')) { // Don't trigger when parcel replies with a default webpage, ie. when the code was not found in the dist folder
        setCode(fetchedCode);
        function scrollToLine(){
          const scrollBox = document.getElementById('code-display');
          const firstLineElem = document.getElementsByClassName('token-line')[0];
          if (scrollBox && props.op.source_file_num && firstLineElem) {
            const lineHeight = firstLineElem.offsetHeight;
            scrollBox.scrollTop = lineHeight * props.op.source_file_num; // Multiply the line number by the height of each line
          }
        }
        const timer = setTimeout(scrollToLine,25);
      } else {
        console.log('Failed to find source code file at: ', sourceUrl);
      }
    })
  });

  return (
    <div className='opacity-80'>
      { code &&
        <div id="code-display" className='whitespace-pre-line border-solid border-2 rounded border-neutral-300'
          style={{
            maxHeight: '200px',
            overflowY: 'scroll',
          }}
        >
          <SyntaxHighlight theme={theme} code={code} highlightLine={props.op.source_file_num} language="python" />
        </div>
      }
      { (code && model.total_accuracy_str) &&
        <div>This workload used additional manual annotations to measure that Hotline's automatic annotation was {model.total_accuracy_str} accurate.</div>
      }
    </div>
  )
}

function MetadataTable(props) {
  if (!props.op) {
    return
  }
  const Container = styled(TableContainer)(({ theme }) => ({
    padding: theme.spacing(2),
    marginBottom: theme.spacing(2),
  }));

  const Row = styled(TableRow)(({ theme }) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: `1px solid ${theme.palette.divider}`,
    '&:last-child': {
      borderBottom: 'none',
    },
  }));

  const Cell = styled(TableCell)(({ theme }) => ({
    padding: theme.spacing(1, 2),
  }));

  // Convert to format: [{ name: 'a', value: 1 }, { name: 'b', value: 2 }
  const flattenedOp = flattenObject(props.op);
  const rows = Object.keys(flattenedOp).map((key) => {
    let value = JSON.stringify(flattenedOp[key]);
    if (typeof value === 'string' || value instanceof String) {
      value = value.replace(/(^"|"$)/g, ''); // remove double quotes from start and end of a string
    }
    if (value === undefined) {
      value = 'undefined';
    }
    const row = { name: key, value: value };
    return row;
  });

  return (
    <div className='opacity-80 my-2 border-2 rounded border-neutral-300'>
      <Container sx={{ maxHeight: 305, padding: 0, marginBottom: 0 }}>
        <Table >
          <TableHead>
              <Row>
                <Cell className='bg-gray-100' sx={{ width: '50%', fontWeight: "bold" }}>Key</Cell>
                <Cell className='bg-gray-100' sx={{ width: '50%', fontWeight: "bold" }}>Value</Cell>
              </Row>
            </TableHead>
          <TableBody>
            {rows.map((row) => (
              <Row key={row.name}>
                <Cell sx={{ width: '50%' }}>{row.name}</Cell>
                <Cell sx={{ width: '50%', textAlign: 'left' }}>{row.value}</Cell>
              </Row>
            ))}
          </TableBody>
        </Table>
      </Container>
    </div>
  );
}

function loadLocalResults() {
  const dispatch = useDispatch();

  // Load default model
  useEffect(() => {
    const results = import('./results/*.js');
    const modelChoices = Object.keys(results);
    const defaultChoice = Object.keys(results)[0];
    // const defaultChoice = Object.keys(results)[modelChoices.length -1];
    // const defaultChoice = 'ResNet50';

    console.log('modelChoices:', modelChoices);
    console.log('defaultChoice: ', defaultChoice);
    if (!defaultChoice) {
      return
    }
    results[defaultChoice]()
    .then((imported) => {
      const model = imported.model[0];
      if (model["metadata.batch_size"]) {
        model["metadata.batch_size"] = Number(model["metadata.batch_size"]).toLocaleString(); // convert 1000 to 1,000
      }
        dispatch(setModel(model));
        dispatch(selectClick({op: model}));
    });
    for (const key in results) {
      results[key]()
      .then((imported) => {
        const model = imported.model[0];
        if (model["metadata.batch_size"]) {
          model["metadata.batch_size"] = Number(model["metadata.batch_size"]).toLocaleString(); // convert 1000 to 1,000
        }
        var workload = {
          model: model,
          id: model.id,
          model_name: model["metadata.model"],
          dataset: model["metadata.dataset"],
          optimizer: model["metadata.optimizer"],
          batch: model["metadata.batch_size"],
          gpu: model["gpu_model"],
          runtime: model["runtime_str"],
        }
        dispatch(addWorkloadOption(workload));
      });
    }
  }, []); // Call once on first render
}

function loadGitHubResults() {
  const dispatch = useDispatch();
  // Load default model
  useEffect(() => {
    const resultUrl = githubBaseUrl + githubResultsList[0];

    // Load workloads
    githubResultsList.forEach((item) => {
      const workloadUrl = githubBaseUrl + item;
      console.log(workloadUrl);

      // Load model
      fetch(workloadUrl)
      .then(response => response.json())
      .then((resp) => {
        const result = resp[0];

        if (result["metadata.batch_size"]) {
          result["metadata.batch_size"] = Number(result["metadata.batch_size"]).toLocaleString(); // convert 1000 to 1,000
        }
        var workload = {
          model: result,
          id: result.id,
          model_name: result["metadata.model"],
          dataset: result["metadata.dataset"],
          optimizer: result["metadata.optimizer"],
          batch: result["metadata.batch_size"],
          gpu: result["gpu_model"],
          runtime: result["runtime_str"],
        }
        dispatch(addWorkloadOption(workload));
      });
    });

    // Load model
    fetch(resultUrl)
    .then(response => response.json())
    .then((resp) => {
      const result = resp[0];

      if (result["metadata.batch_size"]) {
        result["metadata.batch_size"] = Number(result["metadata.batch_size"]).toLocaleString(); // convert 1000 to 1,000
      }
      dispatch(selectClick({op: result}));
      dispatch(setModel(result));
    });


  }, []); // Call once on first render
}

export function App(props) {
  const dispatch = useDispatch();

  if (loadMode === 'github') {
    loadGitHubResults();
  } else {
    loadLocalResults();
  }

  const model = useSelector((state) => state.model.value)
  if (!model) {
    return
  }
  dispatch(displayNextLevel({name: model.name, idx: 1, ops: model.ops, selected: null, op: model}))

  return (
    <div id='hotline-app' className='bg-[#ebebeb]'
      style={{
        lineHeight: 1.5,
        fontFamily: 'system-ui, sans-serif',
        textDecoration: 'none',
      }}
    >
      <Helmet>
        <title>Hotline</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔥</text></svg>"></link>
      </Helmet>

      {/* <div class="flex w-full">
        <div class="w-[10%] bg-gray-600 text-white">
          <div id='left-help-panel' className=''>
            Left
          </div>
        </div>
        <div class="w-[80%]"> */}
          <Header/>
          <Viewer/>
          <Details/>
          <MetadataTable/>
          <Links/>
        {/* </div>
        <div class="w-[10%] bg-gray-600 text-white">
          <div className=''>
            Left
          </div>
        </div>
      </div> */}


      <Shortcuts/>
      <Listeners/>
    </div>
  );
}
