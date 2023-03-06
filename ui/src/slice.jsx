import React, {useState, useEffect, useLayoutEffect} from 'react';
import { useSelector,  useDispatch } from 'react-redux';
import { displayNextLevel, sliceHighlight } from './redux/levelsSlice';
import { selectClick, setWidth } from './redux/appSlice';
import { checkIsOpSelected, checkIsSearched, getSubSlices } from './utils';
import LeaderLine from 'react-leader-line/leader-line.min';
import * as C from './constants_';
let colormap = require('colormap')


function Slice(props) {
  const uid = `${props.resourceName}-${props.op.id}`;
  const labelId = `label-level${props.levelIdx}-${props.resourceName}${props.sliceIdx}`;
  const textId = `text-${props.resourceName}-${props.op.id}`;
  const res_pretty_name = 'res_name' in props.op.resources[props.resourceName] ? props.op.resources[props.resourceName]['res_name'] : props.op.pretty_name;
  const dispatch = useDispatch()

  var trackWidthAdjustment = 100;
  if (props.op.fused_ids) {
    trackWidthAdjustment = 95;
  }


  var sliceCss = {
    position: 'relative',
    height: '3.3rem',
    width: `calc(${props.time.relative_dur * trackWidthAdjustment}% - 2px)`,  // -2px because the outline causes 0% width slices to take up space
    // Default outline style
    '&:before': {
      borderWidth: '2px 1px',
      borderStyle: 'solid',
      borderColor: '#555',
      position: 'absolute',
      content: "''",
      height: '100%',
      width: '100%',
    },
  };

  const colorize = useSelector((state) => state.view.colorize)
  if (colorize) {
    var nshades = 100;
    var colors = colormap({
      colormap: 'hot', // Hot line goes bling
      // colormap: 'inferno', //     var nshades = 200;
      // colormap: 'viridis',
      // colormap: 'rainbow',  // var nshades = 200; reverse method #1
      // colormap: 'blackbody',
      // colormap: 'RdBu',
      // colormap: 'autumn',
      // colormap: 'temperature',
      // colormap: 'freesurface-red',
      nshades: nshades,
      format: 'hex',
    })
    var color_idx = nshades-Math.round(props.time.relative_dur*100);
    // var color_idx = nshades-Math.round(101-(props.time.relative_dur*100));  // reverse method #1
    // colors = colors.reverse();  // reverse method #2

    sliceCss.backgroundColor = colors[color_idx]+'70' // Add for alpha

    // Increase color intensity on mouse over
    const offset = Math.max(color_idx-10, 0);
    sliceCss['&:hover'] = {
      color: offset < 28 ? 'white' : 'black',
      backgroundColor: colors[offset],
    }

  }


  if (props.time.relative_gap_to_previous) {
    sliceCss.marginLeft = (props.time.relative_gap_to_previous * trackWidthAdjustment) + '%';
  }

  // Show next level on click
  const onClick = () => {
    dispatch(selectClick({op: props.op, resourceName: props.resourceName}));

    dispatch(displayNextLevel({
      name: props.op.pretty_name,
      idx: parseInt(props.levelIdx) + 1,
      ops: props.op.ops,
      op: props.op,
      selected: null
    }));

    dispatch(sliceHighlight({
      idx: props.levelIdx,
      selected: props.op
    }));

    // Trigger re-render of hotline timeline because of a rare bug that moves all leader lines 5 px to the left, such as when clicking on the "aten::nll_loss_nd" operation. Note when the bug is not present, this the user doesn't see anything.
    setTimeout(() => {
      var elem = document.getElementById('level-1-track-0');
      if (elem) {
        dispatch(setWidth(elem.clientWidth - Math.random()));
      }
    }, 5);

  };

  // outline if searched
  const searchedVal = useSelector((state) => state.search.value);
  const sliceNames  = getSubSlices(props.op, 'name');
  const sliceIds  = getSubSlices(props.op, 'id');
  const isSearched = searchedVal && checkIsSearched(
    searchedVal, [props.op.pretty_name, res_pretty_name], sliceNames.concat(sliceIds)
  );
  if (isSearched) {
    sliceCss.boxShadow = 'inset 0px 0px 0px 2px red';
    if (props.labelsOnTop) {
      sliceCss.animation = 'bounce 1s 2';
    } else {
      sliceCss.animation = 'bounceDown 1s 2';
    }
  }

  // Slice outline style if clicked on
  const selectedOp = useSelector((state) => state.levels.value[props.levelIdx].selected)
  var isSelected = checkIsOpSelected(props.op, selectedOp);
  if (isSelected) {
    sliceCss['&:before'].borderStyle = 'dotted';
    sliceCss['&:before'].borderWidth = '2px 2px';
  }

  // if op has no sub_ops, animate when clicked
  sliceCss[':active'] = {
    borderStyle: 'dotted',
    transform: 'translateY(4px)'
  };

  // Handle window resize events
  const currentWidthPx = useSelector((state) => state.width.value)
  const [isOverflow, setIsOverflow] = useState(false);
  function onResize() {
    const elem = document.getElementById(textId);
    if (!elem) {
      return
    }
    const fontSizeMultiple = C.fontSizeMultiple;
    const thisLabelWidthPx = res_pretty_name.length * fontSizeMultiple;
    const thisSliceWidthPx = props.time.relative_dur * currentWidthPx;
    const isWideSlice = thisSliceWidthPx > C.wideSliceLenThresh;
    const isOverflow = thisLabelWidthPx >= thisSliceWidthPx && !isWideSlice;
    setIsOverflow(isOverflow);
  }
  React.useLayoutEffect(onResize,[currentWidthPx]); // Call once on first render

  const [isLabelHover, setIsLabelHover] = useState(false);
  var overflowLabel = null;

  // Create leader line label if overflow
  if (isOverflow) {
    const {x, y, abbreviate} = props.getOffset();
    // console.log('abbreviate: ', abbreviate);
    var labelCss = {}
    if (isSelected) {
      labelCss.textDecoration = 'underline';
      labelCss.textDecorationStyle = 'dotted';
      labelCss.textDecorationtThickness = '2px';
    }
    if (abbreviate && !isLabelHover) {
      labelCss.width = `${abbreviate}px`;
      labelCss.position = 'absolute';
      labelCss.overflow = 'hidden';
      labelCss.textOverflow = 'ellipsis';
      labelCss.top = '-0.5em';
    }
    overflowLabel = (
      <div id={labelId} className="font-mono underline-offset-2"
        onMouseEnter={() => setIsLabelHover(true)}
        onMouseLeave={() => setIsLabelHover(false)}
      style={{transform: `translate(${x}px, ${y}px)`,
        position: 'absolute',
        left: '50%',
        whiteSpace: 'nowrap'
      }}>
        <span className='pl-1 pb-1' style={{ ...labelCss}}>
          {res_pretty_name}
        </span>
      </div>)
  }

  const [isHover, setIsHover] = useState(false);
  // If op is wide enough to display some but not all text
  // const longName =  res_pretty_name.length > 150  // TODO(deferred) instead of 150 letters, check if the name overflows the inner area of the slice element
  const longName = false; // Disabled

  const multiLineTextClampCss = {
    overflow: 'hidden',
    display: '-webkit-box',
    WebkitBoxOrient: 'vertical',
    WebkitLineClamp: '3',
  }
  var isHoverCss = {
    zIndex: '1'
  };
  if (isHover && longName) {
    isHoverCss.backgroundColor = 'white';
    isHoverCss.zIndex = '9';
    isHoverCss.position = 'relative';
    isHoverCss.borderWidth = '2px 1px';
    isHoverCss.borderColor = '#555';
    isHoverCss.borderStyle = 'solid';
    if (isSelected) {
      isHoverCss.borderStyle = 'dotted';
      isHoverCss.borderWidth = '2px 2px';
    }
  }
  const pretty_name_html = <span dangerouslySetInnerHTML={{__html: res_pretty_name}} />;  // TODO: Don't do, this can be exploited easily

  // Draw leader line if overflow
  useLayoutEffect(() => {
    if (isOverflow && document.getElementById(labelId)){
        var attachOffsetStart = props.labelsOnTop ? '0%' : '100%';
        var attachOffsetEnd = props.labelsOnTop ? '50%' : '50%';
        const start = LeaderLine.pointAnchor({element: document.getElementById(uid), x:'50%', y: attachOffsetStart});
        const end = LeaderLine.pointAnchor({element: document.getElementById(labelId), x:'0%', y: attachOffsetEnd});
        const line = new LeaderLine({
            start: start,
            end: end,
            path: 'straight',
            startPlug: 'behind',
            endPlug: 'square',
            size: 1,
            color: 'grey',
        }
        );
      return function cleanup() {
        line.remove();
      }
    }
  });

  return (
    <span id={uid}
      className="float-left flex justify-center items-center leading-4 cursor-pointer"
      onClick={onClick}
      css={sliceCss}
      onMouseEnter={() => setIsHover(true)}
      onMouseLeave={() => setIsHover(false)}
    >
      <span id={textId} className={'font-mono text-center text-clip break-word overflow-hidden ' + (isOverflow && 'hidden')} style={isHoverCss} >
          { (longName && !isOverflow && !isHover)
            ?
              <span style={multiLineTextClampCss}>
                {pretty_name_html}
              </span>
            :
            <span>
              {pretty_name_html}
            </span>
          }
      </span>
      { isOverflow && overflowLabel }
    </span>
  );
}

export default Slice

