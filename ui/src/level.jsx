import React, {useState, useEffect, useLayoutEffect} from 'react';
import { useSelector,  useDispatch } from 'react-redux';
import { selectClick } from './redux/appSlice';
import { checkLabelsOnTop, checkIsOpSelected, checkIsStartAdjusted } from './utils';
import Slice from './slice';
import * as C from './constants_';


function Level(props) {
  if (!('ops' in props && props.ops)) {
    return
  }

  const summarizeResources = useSelector((state) => state.view.summarizeResources);
  // const summarizeResources = false;  // uncomment to see all resources by default
  var resourceSlicesObj = {};
  for (const op of props.ops) {
    for (const resourceName of Object.keys(op.resources)) {
      var time = op.resources[resourceName]['time'];
      if (summarizeResources && 'parent_is_longest' in time && !time.parent_is_longest) {
        continue // Exclude resources that are not the longest
      }
      if (!(resourceName in resourceSlicesObj)) {
        resourceSlicesObj[resourceName] = {}
        resourceSlicesObj[resourceName]['slices'] = [];
      }
      const labelsOnTop = checkLabelsOnTop(resourceName);
      var sliceObj = {
        key: op.id,
        op: op,
        time: time,
        levelIdx: props.levelIdx,
        resourceName: resourceName,
        labelsOnTop: labelsOnTop
      }
      resourceSlicesObj[resourceName]['slices'].push(sliceObj);
    }
  }

  const currentWidthPx = useSelector((state) => state.width.value);
  if (currentWidthPx == 0) {
    return
  }

  var resourceSlices = {};
  var resourceLabelPadding = {};
  for (let [resourceName, res] of Object.entries(resourceSlicesObj)) {
    let slices = res['slices'];
    if (!(resourceName in resourceSlices)) {
      resourceSlices[resourceName] = {};
      resourceSlices[resourceName]['slices'] = [];
      resourceLabelPadding[resourceName] = 0;
    }

    // BEGIN Calculate label Y offset and right abbriviation
    const labelsOnTop = checkLabelsOnTop(resourceName);
    var lastLabelStart = {};
    var offsetY = C.offsetYInitial;
    const relativeDurSum = slices.map(slice => slice.time.relative_dur + (slice.time.relative_gap_to_previous ? slice.time.relative_gap_to_previous : 0)).reduce((a,b)=>a+b)
    var pxToRightBorder = (1 - relativeDurSum) * (currentWidthPx - 33 )
    var pxToRightBorder2 = (1 - relativeDurSum) * (currentWidthPx - 33 )
    var pxAccumulator = 0;  // Accumulates the width of slices where 0 the left edge of the rightmost slice and each increment jumps to the next self edge of a slice.
    for (const [sliceIdx, slice] of slices.reverse().entries()) {
      const labelName = 'res_name' in slice.op.resources[resourceName] ? slice.op.resources[resourceName]['res_name'] : slice.op.pretty_name;
      const relativeDur = slice.op.resources[resourceName]['time'].relative_dur;
      const relativeGap = slice.op.resources[resourceName]['time'].relative_gap_to_previous ? slice.op.resources[resourceName]['time'].relative_gap_to_previous : 0;
      const thisLabelWidthPx = labelName.length * C.fontSizeMultiple;
      const thisSliceWidthPx = relativeDur * currentWidthPx;
      const thisSliceGapPx = relativeGap * currentWidthPx;
      const isWideSlice = thisSliceWidthPx > C.wideSliceLenThresh;
      pxAccumulator += thisSliceWidthPx;
      pxAccumulator += thisSliceGapPx;
      pxToRightBorder += thisSliceWidthPx - 3;
      pxToRightBorder2 += thisSliceGapPx + thisSliceWidthPx - 3;
      const labelRightEdgePx = pxToRightBorder + (thisSliceWidthPx/2) - thisLabelWidthPx ;  // 0 at right border of track, max at left side of track
      const labelRightEdgePosPx = (pxAccumulator - thisSliceWidthPx) + thisLabelWidthPx
      if (labelRightEdgePosPx >= pxAccumulator) {
        const isOverflow = thisLabelWidthPx >= thisSliceWidthPx && !isWideSlice;
        if (isOverflow) {
          offsetY += C.offsetYIncrement;
          var abbreviate = false;
          // console.log(`${resourceName} ${labelName} ${labelRightEdgePx}`)
          if (labelRightEdgePx < 50 )  // if close enough to right border
          {
            const offsetYIsNew = !Object.keys(lastLabelStart).includes(offsetY.toString());
            const maxOffset = Math.max.apply(Math, Object.keys(lastLabelStart));
            const lastStartAbbreviation = lastLabelStart[maxOffset] + 50
            const lastStartAbbreviationNotNull = lastStartAbbreviation && !offsetYIsNew ? lastStartAbbreviation : 0
            abbreviate = pxToRightBorder2 + 40 - (thisSliceWidthPx / 2) - lastStartAbbreviationNotNull;
          }
        }
      } else {
        offsetY = C.offsetYInitial;
      }
      const isOverflow = thisLabelWidthPx >= thisSliceWidthPx && !isWideSlice;
      if (isOverflow) {
        const wtf = offsetY === C.offsetYInitial ? C.offsetYInitial + C.offsetYIncrement : offsetY;
        lastLabelStart[wtf] = pxAccumulator - (thisSliceWidthPx / 2);
      }


      const retAbbreviate = abbreviate;
      const curoffsetY = offsetY;
      function getOffSet() {
        const retoffsetY = curoffsetY === C.offsetYInitial ? C.offsetYInitial + C.offsetYIncrement : curoffsetY;
        return {
          x: 0,
          y: retoffsetY * (labelsOnTop ? -1 : 1),
          abbreviate: retAbbreviate,
        };
      }
      // END Calculate label Y offset
      const slice_component = <Slice {...slice} sliceIdx={sliceIdx} getOffset={getOffSet}></Slice>;
      resourceSlices[resourceName]['slices'].unshift(slice_component);
      resourceLabelPadding[resourceName] = Math.max(resourceLabelPadding[resourceName], offsetY);
    }
  }

  const resourceFilter = useSelector((state) => state.view.resourceFilter);
  if (resourceFilter !== 'All') {
    for (const [key, value] of Object.entries(resourceSlices)) {
      const filter_match = key.toLowerCase().includes(resourceFilter.toLowerCase());
      if (!filter_match) {
        delete resourceSlices[key];
      }
    }
  }

  return (
  <div>
    <div className={'' + (props.levelIdx != 1 ? ' border-t-[3px] border-neutral-900' : '')}>
      <div className='flex w-full'>
        <div className='w-[20%] flex justify-center items-center text-center'>
          <LevelTitle op={props.op} levelIdx={props.levelIdx} name={props.name} />
        </div>
        <div className='w-[80%]'>
          {Object.keys(resourceSlices).sort().map((resourceName, resourceIdx) =>
            <ResourceTrack key={resourceName + props.levelIdx} slices={resourceSlices[resourceName]['slices']} time={props.op.resources[resourceName]['time']} resourceName={resourceName} levelIdx={props.levelIdx} resourceIdx={resourceIdx} labelPadding={resourceLabelPadding[resourceName]} />
          )}
        </div>
      </div>
    </div>
  </div>
  )
}

function LevelTitle(props) {
  const dispatch = useDispatch()
  // Show next level on click
  const onClick = () => {
    dispatch(selectClick({op: props.op}));
  };

  // Check if is selected
  const selectedOp = useSelector((state) => state.select.value)
  var isSelected = checkIsOpSelected(props.op, selectedOp);
  const levelTitleCss = {}
  if (isSelected) {
    levelTitleCss.textDecoration = 'underline';
    levelTitleCss.textDecorationStyle = 'dotted';
    levelTitleCss.textUnderlineOffset = '5px';
  }

  return (
    <div className='text-lg mx-2 flex flex-full w-full'>
      <div className='w-[20%] grid place-content-center'>
        <span className='levelNumberCircle'>
          <span className='levelNumberText'>
            {props.levelIdx}
          </span>
        </span>
      </div>
      <span className='text-neutral-900 w-[80%] overflow-hidden cursor-pointer' onClick={onClick} style={{ ...levelTitleCss}}>
        {props.name}
      </span>
    </div>
  )
}

function ResourceTrack(props) {
  const labelsOnTop = checkLabelsOnTop(props.resourceName);
  const labelPadding = (props.labelPadding-10).toString() + 'px';
  var resourceTitleClasses = '';
  var css = {};
  if (labelsOnTop) {
    css.paddingTop = labelPadding;
    resourceTitleClasses = 'underline decoration-2 decoration-sky-300 underline-offset-4';
  } else {
    css.paddingBottom = labelPadding;
    resourceTitleClasses = 'underline decoration-2 decoration-green-300 underline-offset-4';
  }
  const trackId = `level-${props.levelIdx}-track-${props.resourceIdx}`;

  const isStartAdjusted = checkIsStartAdjusted(props);
  const startAdjustmentStyle = {
    position: 'absolute',
    left: '-1px',
    top: !labelsOnTop ? '1.4rem' : '',
    bottom: labelsOnTop ? '1.4rem' : '',
    color: 'rgb(212 212 212)',
    display: isStartAdjusted ? '' : 'none',
  };
  return (
    <div>
    { props.resourceIdx>0 &&
      <div className='border-t-2 border-neutral-400'></div>
    }
    <div className='flex w-full'>
      <div className='w-[90%] flex justify-center items-center border-l-2 border-neutral-300 relative pl-[7px]'>
        <i className="bi-activity absolute" style={startAdjustmentStyle}></i>
        <div className="p-2 pr-6 inline-block w-full truncate " css={css} id={trackId}>
          {/* Reminder: overflow hidden due to truncate className */}
          {props.slices}
        </div>
      </div>
      <div className='w-[10%] flex text-center justify-center items-center border-l-2 border-neutral-400'>
        <div>
          <div className={'text-lg text-neutral-900 uppercase ' + resourceTitleClasses}>
            {props.resourceName}
          </div>
          <div className='text-neutral-500'>
            {props.time['runtime_str']}
          </div>
        </div>
      </div>
    </div>
  </div>
  )
}

export default Level