
import styled from "styled-components";
import Highlight, { defaultProps } from "prism-react-renderer";
// Docs: https://github.com/FormidableLabs/prism-react-renderer
// Languages supported: https://github.com/FormidableLabs/prism-react-renderer/blob/master/src/vendor/prism/includeLangs.js

const Pre = styled.pre`
  text-align: left;
  // margin: 1em 0;
  // padding: 0.5em;
  // overflow: scroll;
`;

const Line = styled.div`
  display: table-row;
`;

const LineNo = styled.span`
  display: table-cell;
  text-align: right;
  padding-right: 1em;
  user-select: none;
  opacity: 0.5;
`;

const LineContent = styled.span`
  display: table-cell;
`;

function SyntaxHighlight(props) {
  let style = {
    padding: '10px',
  };
  return (
    <div className="SyntaxHighlight" style={style}>
      <Highlight
        {...defaultProps}
        theme={props.theme}
        code={props.code}
        language={props.language}
      >
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <Pre className={className}
            style={{ fontFamily: "'fontFamily.mono', SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace"}}
          >
            {tokens.map((line, i) => {
              const lineProps = getLineProps({ line, key: i });
              lineProps.style = {
                display: 'block',
              };
              if (i === props.highlightLine-1) {
                lineProps.style.backgroundColor = "rgb(122 200 230 / 27%)" // light blue
                // lineProps.style.backgroundColor = "#e6cf7a45" // light yellow
              }
              return (
                <Line key={i} {...lineProps}>
                  <LineNo>{i + 1}</LineNo>
                  <LineContent>
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token, key })} />
                    ))}
                  </LineContent>
                </Line>
              );
            })}
          </Pre>
        )}
      </Highlight>
    </div>
  );
}

export default SyntaxHighlight;